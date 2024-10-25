import argparse
import json
import logging
from pathlib import Path
from typing import Any

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


def main(args: Any) -> None:
    """Main function to train a StaticModel."""
    logging.basicConfig(level=logging.INFO)

    # Use distill or random embeddings based on CLI args
    if args.random_embeddings:
        logging.info("Using random embeddings.")
        s = distill(args.model_name)
        v = np.random.randn(*s.embedding.shape)  # noqa NPY002
        v = _post_process_embeddings(v, "auto", False).astype(np.float32)
        s = StaticModel(v, s.tokenizer)
    else:
        s = distill(args.model_name)

    # Collect paths for training and validation
    paths = sorted(Path(args.data_path).glob("*.json"))
    train_paths, test_paths = paths[:-1], paths[-1:]  # Last one used for validation/test

    train_txt, train_vec = collect_means_and_texts(train_paths)
    val_txt, val_vec = collect_means_and_texts(test_paths)

    train_data = TextDataset(train_txt, torch.from_numpy(train_vec), s.tokenizer)
    val_data = TextDataset(val_txt, torch.from_numpy(val_vec), s.tokenizer)

    # Train the model
    model, _ = train_supervised(train_data, val_data, s, device=args.device)
    # Save the trained model
    model.save_pretrained(args.save_path)


if __name__ == "__main__":
    # Define CLI arguments
    parser = argparse.ArgumentParser(description="Train StaticModel with distillation and dataset paths.")

    parser.add_argument(
        "--model-name",
        type=str,
        default="baai/bge-base-en-v1.5",
        help="The model name for distillation (e.g., 'baai/bge-base-en-v1.5').",
    )
    parser.add_argument(
        "--data-path", type=str, default="data/fineweb_bgebase", help="Path to the directory containing the dataset."
    )
    parser.add_argument("--save-path", type=str, help="Path to save the trained model.")

    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run the training on (e.g., 'cpu', 'cuda')."
    )
    parser.add_argument(
        "--random-embeddings", action="store_true", help="Use random embeddings instead of distilling the model."
    )

    args = parser.parse_args()

    # Run main function
    main(args)
