import json
import logging
from pathlib import Path

import numpy as np
import torch
from model2vec import StaticModel
from model2vec.distill import distill
from model2vec.distill.distillation import _post_process_embeddings
from reach import Reach

from nanofit.train import TextDataset, train_supervised


def collect_means_and_texts(paths: list[Path]) -> tuple[list[str], np.ndarray]:
    """Collect means and texts from a bunch of reach paths."""
    txts = []
    v = []
    for path in paths:
        if not path.name.endswith(".json"):
            continue
        try:
            r = Reach.load(path)
        except KeyError:
            # Workaround for old format reach
            # whatever
            vectors_path = str(path).replace("_items.json", "_vectors.npy")
            items = json.load(open(path))["items"]
            vectors = np.load(open(vectors_path, "rb"))
            r = Reach(vectors, items)
        # Filter out any NaN vectors before appending
        non_nan_indices = ~np.isnan(r.vectors).any(axis=1)
        valid_vectors = r.vectors[non_nan_indices]
        valid_items = np.array(r.sorted_items)[non_nan_indices]
        txts.extend(valid_items)
        v.append(valid_vectors)

    return txts, np.concatenate(v)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    s = distill("baai/bge-base-en-v1.5", pca_dims="auto", apply_zipf=False)
    v = np.random.randn(*s.embedding.shape)  # noqa
    v = _post_process_embeddings(v, "auto", False).astype(np.float32)

    s = StaticModel(v, s.tokenizer)

    paths = sorted(Path("data/c4_old").glob("*.json"))
    train_paths, test_paths = paths[:-1], paths[-1:]
    paths = sorted(Path("data/fineweb").glob("*.json"))
    train_paths.extend(paths[:-1])
    test_paths.extend(paths[-1:])

    train_txt, train_vec = collect_means_and_texts(train_paths)
    val_txt, val_vec = collect_means_and_texts(train_paths)

    train_data = TextDataset(train_txt, torch.from_numpy(train_vec), s.tokenizer)
    val_data = TextDataset(val_txt, torch.from_numpy(val_vec), s.tokenizer)

    model, trained_model = train_supervised(train_data, val_data, s, device="cpu", batch_size=256, patience=3)
