import logging
from pathlib import Path

import numpy as np
import torch
from model2vec import distill
from reach import Reach

from nanofit.train import TextDataset, train_supervised


def collect_means_and_texts(paths: list[Path]) -> tuple[list[str], np.ndarray]:
    """Collect means and texts from a bunch of reach paths."""
    txts = []
    v = []
    for path in paths:
        r = Reach.load(path)
        txts.extend(r.sorted_items)
        v.append(r.vectors)

    return txts, np.concatenate(v)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    s = distill("baai/bge-base-en-v1.5", pca_dims=None, apply_zipf=False)
    s.embedding_bag.requires_grad_(True)

    w = s.embedding.weight.clone().detach()

    paths = sorted(Path("fineweb").glob("*.json"))
    train_paths, test_paths = paths[:-2], paths[-2:]

    train_txt, train_vec = collect_means_and_texts(train_paths)
    val_txt, val_vec = collect_means_and_texts(train_paths)

    train_data = TextDataset(train_txt, torch.from_numpy(train_vec), s.tokenizer)
    val_data = TextDataset(val_txt, torch.from_numpy(val_vec), s.tokenizer)

    model = train_supervised(train_data, val_data, s)
