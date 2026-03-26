import argparse
import logging
import shutil
from pathlib import Path
from typing import Iterator

import numpy as np
from datasets import Dataset, Features, Sequence, Value, concatenate_datasets, load_dataset, load_from_disk
from huggingface_hub import DatasetCard, DatasetCardData
from more_itertools import batched
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer

_DATASET_CARD_TEMPLATE = Path(__file__).parent / "datacards" / "dataset_card_template.md"

_SAVE_EVERY = 1024  # rows (≈ 32 batches at the default batch size of 32)

_FEATURES = Features({"text": Value("string"), "embedding": Sequence(Value("float32"))})

logger = logging.getLogger(__name__)


def _save_checkpoint(checkpoints_dir: Path, texts: list[str], embeddings: list[np.ndarray], part_idx: int) -> None:
    """Save a checkpoint part as a Parquet file."""
    part = Dataset.from_dict(
        {"text": texts, "embedding": [e.tolist() for e in embeddings]},
        features=_FEATURES,
    )
    part.to_parquet(str(checkpoints_dir / f"shard_{part_idx:08d}.parquet"))


def _compact_checkpoints(checkpoints_dir: Path, output_dir: Path, keep_checkpoints: bool) -> None:
    """Compact checkpoint shards into a single standard HuggingFace dataset."""
    shard_files = sorted(checkpoints_dir.glob("shard_*.parquet"))
    if not shard_files:
        return

    logger.info("Compacting checkpoints into final dataset...")
    # Build the compacted dataset in a sibling temp dir, then replace output_dir.
    tmp_dir = output_dir.parent / f"{output_dir.name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    # Load all shards and concatenate them into a single dataset, then save to the temp dir.
    dataset = concatenate_datasets([Dataset.from_parquet(str(f)) for f in shard_files])
    dataset.save_to_disk(str(tmp_dir))
    if output_dir.exists():
        # Remove the old output dir before renaming the temp dir to avoid leaving stale Arrow files from previous runs.
        shutil.rmtree(output_dir)
    tmp_dir.rename(output_dir)
    if not keep_checkpoints:
        shutil.rmtree(checkpoints_dir)
    logger.info(f"Dataset saved to {output_dir}")


def _create_dataset_card(
    output_dir: Path,
    model_name: str,
    source_dataset: str,
    source_name: str,
    source_split: str,
    num_rows: int,
    embedding_dim: int,
    repo_id: str | None = None,
) -> DatasetCard:
    """Create a dataset card, save it to the output directory, and return it."""
    card_data = DatasetCardData(
        language="en",
        tags=["tokenlearn", "embeddings", "model2vec"],
    )
    card = DatasetCard.from_template(
        card_data,
        template_path=str(_DATASET_CARD_TEMPLATE),
        repo_id=repo_id,
        dataset_name=output_dir.name,
        model_name=model_name,
        source_dataset=source_dataset,
        source_name=source_name,
        source_split=source_split,
        num_rows=num_rows,
        embedding_dim=embedding_dim,
    )
    card.save(output_dir / "README.md")
    return card


def featurize(  # noqa C901
    dataset: Iterator[dict[str, str]],
    model: SentenceTransformer,
    output_dir: str,
    max_rows: int,
    batch_size: int,
    text_key: str,
    max_length: int | None = None,
    keep_checkpoints: bool = False,
) -> None:
    """Make a directory and dump all kinds of data in it."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = Path(str(output_dir_path) + ".checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)

    shard_files = sorted(checkpoints_dir.glob("shard_*.parquet"))
    part_idx = len(shard_files)
    rows_done = sum(len(Dataset.from_parquet(str(f))) for f in shard_files)
    if rows_done:
        logger.info(f"Resuming from {rows_done} previously written rows ({part_idx} checkpoint shards).")

    texts = []
    embeddings = []
    dim = model.get_sentence_embedding_dimension()
    if dim is None:
        raise ValueError("Model has no sentence embedding dimension.")

    tokenizer: PreTrainedTokenizer = model.tokenizer
    if max_length is not None:
        # Set both tokenizer and model max length
        tokenizer.model_max_length = max_length
        model.max_seq_length = max_length
        logger.info(f"Set tokenizer maximum length to {max_length}.")
    # Binding i in case the dataset is empty.
    i = 0
    for i, batch in tqdm(enumerate(batched(dataset, n=batch_size))):
        batch = list(batch)
        rows_processed = i * batch_size
        if rows_processed >= max_rows:
            logger.info(f"Reached maximum number of rows: {max_rows}")
            break
        # Skip batches fully covered by a previous run; trim those that straddle the boundary.
        if rows_processed + len(batch) <= rows_done:
            continue
        if rows_processed < rows_done:
            batch = batch[rows_done - rows_processed :]
        batch = [x[text_key] for x in batch]

        if not all(isinstance(x, str) for x in batch):
            raise ValueError(f"Detected non-string at batch: {i}")

        batch_embeddings = model.encode(batch, output_value="token_embeddings")  # type: ignore  # Annoying
        for text, embedding in zip(batch, batch_embeddings):
            texts.append(_truncate_text(tokenizer, text))
            embeddings.append(embedding[1:-1].float().mean(axis=0).cpu().numpy())
        if len(texts) >= _SAVE_EVERY:
            _save_checkpoint(checkpoints_dir, texts, embeddings, part_idx)
            part_idx += 1
            texts = []
            embeddings = []
    if texts:
        _save_checkpoint(checkpoints_dir, texts, embeddings, part_idx)

    _compact_checkpoints(checkpoints_dir, output_dir_path, keep_checkpoints)


def _truncate_text(tokenizer: PreTrainedTokenizer, text: str) -> str:
    """Truncate text to fit the tokenizer's maximum length."""
    tokens = tokenizer.encode(
        text,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    return tokenizer.decode(tokens, skip_special_tokens=True)


def main() -> None:
    """Main function to featurize texts using a sentence transformer."""
    parser = argparse.ArgumentParser(description="Featurize texts using a sentence transformer.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="baai/bge-base-en-v1.5",
        help="The model name for distillation (e.g., 'baai/bge-base-en-v1.5').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the featurized texts.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="allenai/c4",
        help="The dataset path or name (e.g. 'allenai/c4').",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="en",
        help="The dataset configuration name (e.g., 'en' for C4).",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="The dataset split (e.g., 'train', 'validation').",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_false",
        help="Disable streaming mode when loading the dataset.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=1000000,
        help="The maximum number of rows to featurize.",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="text",
        help="The key of the text field in the dataset to featurize (default: 'text').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size to use for encoding the texts.",
    )
    parser.add_argument("--max-length", type=int, default=None, help="Maximum token length for the tokenizer.")
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep checkpoint parts after compaction (default: delete them).",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="HuggingFace Hub repo ID to push the dataset to after featurizing (e.g., 'username/my-dataset').",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        model_name = args.model_name.replace("/", "_")
        dataset_path = args.dataset_path.replace("/", "_")
        output_dir = f"{model_name}_{dataset_path}_featurized"
    else:
        output_dir = args.output_dir

    model = SentenceTransformer(args.model_name)
    dataset = load_dataset(
        args.dataset_path,
        name=args.dataset_name,
        split=args.dataset_split,
        streaming=args.no_streaming,
    )

    featurize(
        iter(dataset),
        model,
        output_dir,
        args.max_rows,
        args.batch_size,
        args.key,
        max_length=args.max_length,
        keep_checkpoints=args.keep_checkpoints,
    )

    output_dir_path = Path(output_dir)
    if (output_dir_path / "dataset_info.json").exists():
        ds = load_from_disk(output_dir)
        card = _create_dataset_card(
            output_dir=output_dir_path,
            model_name=args.model_name,
            source_dataset=args.dataset_path,
            source_name=args.dataset_name,
            source_split=args.dataset_split,
            num_rows=len(ds),
            embedding_dim=len(ds[0]["embedding"]),
            repo_id=args.push_to_hub,
        )
        if args.push_to_hub:
            logger.info(f"Pushing dataset to Hub: {args.push_to_hub}")
            ds.push_to_hub(args.push_to_hub)
            card.push_to_hub(args.push_to_hub)
    else:
        logger.warning("No data was written — skipping dataset card and Hub push.")


if __name__ == "__main__":
    main()
