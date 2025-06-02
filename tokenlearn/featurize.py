import argparse
import json
import logging
from pathlib import Path
from typing import Iterator

import numpy as np
from datasets import load_dataset
from more_itertools import batched
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer

_SAVE_EVERY = 32


logger = logging.getLogger(__name__)


def featurize(
    dataset: Iterator[dict[str, str]],
    model: SentenceTransformer,
    output_dir: str,
    max_means: int,
    batch_size: int,
    text_key: str,
) -> None:
    """Make a directory and dump all kinds of data in it."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Ugly hack
    largest_batch = max([int(x.stem.split("_")[1]) for x in list(output_dir_path.glob("*.json"))], default=0)
    if largest_batch:
        logger.info(f"Resuming from batch {largest_batch}, skipping previous batches.")

    texts = []
    embeddings = []
    dim = model.get_sentence_embedding_dimension()
    if dim is None:
        raise ValueError("Model has no sentence embedding dimension.")

    tokenizer: PreTrainedTokenizer = model.tokenizer
    # Binding i in case the dataset is empty.
    i = 0
    for i, batch in tqdm(enumerate(batched(dataset, n=batch_size))):
        if i * batch_size >= max_means:
            logger.info(f"Reached maximum number of means: {max_means}")
            break
        if largest_batch and i <= largest_batch:
            continue
        batch = [x[text_key] for x in batch]

        if not all(isinstance(x, str) for x in batch):
            raise ValueError(f"Detected non-string at batch: {i}")

        batch_embeddings = model.encode(batch, output_value="token_embeddings")  # type: ignore  # Annoying
        for text, embedding in zip(batch, batch_embeddings):
            texts.append(_truncate_text(tokenizer, text))
            embeddings.append(embedding[1:-1].float().mean(axis=0).cpu().numpy())
        if i and i % _SAVE_EVERY == 0:
            json.dump(texts, open(output_dir_path / f"feature_{i}.json", "w"), indent=4)
            np.save(output_dir_path / f"feature_{i}.npy", embeddings)
            texts = []
            embeddings = []
    if texts:
        json.dump(texts, open(output_dir_path / f"feature_{i}.json", "w"), indent=4)
        np.save(output_dir_path / f"feature_{i}.npy", embeddings)


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
        "--max-means",
        type=int,
        default=1000000,
        help="The maximum number of mean embeddings to generate.",
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

    featurize(iter(dataset), model, output_dir, args.max_means, args.batch_size, args.key)


if __name__ == "__main__":
    main()
