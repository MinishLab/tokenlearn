import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from datasets import load_dataset
from more_itertools import batched
from reach import Reach
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

_SAVE_INTERVAL = 10
_MAX_MEANS = 1100000


def featurize(texts: Iterable[str], model: SentenceTransformer, output_dir: str) -> None:
    """
    Featurize text using a sentence transformer.

    :param texts: Iterable of texts to featurize.
    :param model: SentenceTransformer model to use.
    :param output_dir: Directory to save the featurized texts.
    :raises ValueError: If the model does not have a fixed dimension.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_dim = model.get_sentence_embedding_dimension()
    if model_dim is None:
        raise ValueError("Model does not have a fixed dimension")

    txts = []
    means = []
    seen = set()
    total_means = 0

    for index, batch in enumerate(tqdm(batched(texts, 32))):
        i = index // _SAVE_INTERVAL
        if (out_path / f"featurized_{i}.json").exists():
            continue
        # Consume the generator
        list_batch = [x["text"].strip() for x in batch if x.get("text")]

        # Already truncated to model max_length
        tokenized_ids = model.tokenize(list_batch)["input_ids"]
        token_embeddings: list[np.ndarray] = [
            x.cpu().numpy() for x in model.encode(list_batch, output_value="token_embeddings", convert_to_numpy=True)
        ]

        for tokenized_id, token_embedding in zip(tokenized_ids, token_embeddings, strict=True):
            # Truncate to actual length of vectors, remove CLS and SEP.
            text = model.tokenizer.decode(tokenized_id[1 : len(token_embedding) - 1])
            if text in seen:
                continue
            seen.add(text)
            mean = np.mean(token_embedding[1:-1], axis=0)
            txts.append(text)
            means.append(mean)
            total_means += 1

            if total_means >= _MAX_MEANS:
                # Save the final batch and stop
                r = Reach(means, txts)
                r.save(out_path / f"featurized_{(index // _SAVE_INTERVAL)}.json")
                return

        if index > 0 and (index + 1) % _SAVE_INTERVAL == 0:
            r = Reach(means, txts)
            r.save(out_path / f"featurized_{(index // _SAVE_INTERVAL)}.json")
            txts = []
            means = []
            seen = set()
    else:
        if means:
            r = Reach(means, txts)
            r.save(out_path / f"featurized_{(index // _SAVE_INTERVAL)}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Model2Vec using tokenlearn.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="baai/bge-base-en-v1.5",
        help="The model name for distillation (e.g., 'baai/bge-base-en-v1.5').",
    )
    args = parser.parse_args()
    model = SentenceTransformer(args.model_name)
    dataset = load_dataset("allenai/c4", name="en", split="train", streaming=True)
    featurize(dataset, model, "data/c4_bgebase")
