from pathlib import Path
from typing import Iterable

import numpy as np
from datasets import load_dataset
from more_itertools import batched
from reach import Reach
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

_SAVE_INTERVAL = 10


def featurize(texts: Iterable[str], model: SentenceTransformer, output_dir: str) -> list[tuple[str, np.ndarray]]:
    """Featurize text using a sentence transformer."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_dim = model.get_sentence_embedding_dimension()
    if model_dim is None:
        raise ValueError("Model does not have a fixed dimension")

    txts = []
    means = []

    for index, batch in enumerate(tqdm(batched(texts, 32))):
        i = index // _SAVE_INTERVAL
        if (out_path / f"featurized_{i}.json").exists():
            continue
        # Consume the generator
        list_batch = list(batch)

        # Already truncated to model max_length
        tokenized_ids = model.tokenize(list_batch)["input_ids"]
        # Remove texts that have the same representation.

        texts = []
        seen = set()
        for ids, text in zip(tokenized_ids, list_batch):
            tuple_ids = tuple(ids)
            if tuple_ids in seen:
                continue
            seen.add(tuple(ids))
            texts.append(text)

        tokenized_ids = model.tokenize(texts)["input_ids"]

        token_embeddings: list[np.ndarray] = [
            x.cpu().numpy() for x in model.encode(texts, output_value="token_embeddings", convert_to_numpy=True)
        ]

        for tokenized_id, token_embedding in zip(tokenized_ids, token_embeddings, strict=True):
            # Truncate to actual length of vectors, remove CLS and SEP.
            text = model.tokenizer.decode(tokenized_id[1 : len(token_embedding) - 1])
            mean = np.mean(token_embedding[1:-1], axis=0)
            txts.append(text)
            means.append(mean)

        if index > 0 and (index + 1) % _SAVE_INTERVAL == 0:
            r = Reach(means, txts)
            r.save(out_path / f"featurized_{(index // _SAVE_INTERVAL)}.json")
            txts = []
            means = []
    else:
        if means:
            r = Reach(means, txts)
            r.save(out_path / f"featurized_{(index // _SAVE_INTERVAL)}.json")

    return means


if __name__ == "__main__":
    model = SentenceTransformer("baai/bge-base-en-v1.5")

    # use name="sample-10BT" to use the 10BT sample
    fw = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True)
    means = featurize(fw, model, "data/fineweb")
