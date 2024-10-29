import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from model2vec import StaticModel
from model2vec.distill import distill
from model2vec.distill.distillation import _post_process_embeddings
from reach import Reach
from sklearn.decomposition import PCA
from tqdm import tqdm

from tokenlearn.train import TextDataset, train_supervised

logging.basicConfig(level=logging.INFO)


def collect_means_and_texts(paths: list[Path]) -> tuple[list[str], np.ndarray]:
    """Collect means and texts from a list of reach paths."""
    txts = []
    v = []
    for path in tqdm(paths, desc="Collecting means and texts"):
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


def train_model(
    model_name: str, data_path: str, save_path: str, device: str = "cpu", random_embeddings: bool = False
) -> StaticModel:
    """
    Train a tokenlearnn model.

    :param model_name: The sentence transformer model name for distillation.
    :param data_path: Path to the directory containing the dataset.
    :param save_path: Path to save the trained model.
    :param device: Device to run the training on.
    :param random_embeddings: Use random embeddings instead of distilling the model.
    :return: The trained model.
    """
    if random_embeddings:
        logging.info("Using random embeddings.")
        s = distill(model_name)
        v = np.random.randn(*s.embedding.shape)  # noqa NPY002
        v = _post_process_embeddings(v, 256, False).astype(np.float32)
        s = StaticModel(v, s.tokenizer)
    else:
        s = distill(model_name)

    # Collect paths for training
    paths = sorted(Path(data_path).glob("*.json"))
    train_txt, train_vec = collect_means_and_texts(paths)
    train_data = TextDataset(train_txt, torch.from_numpy(train_vec), s.tokenizer)

    # Train the model
    model, _ = train_supervised(train_data, s, device=device)

    # Save the trained model
    model.save_pretrained(save_path)

    return model


def weight_model(model_name: str, data_path: str, pca_dims: int) -> StaticModel:
    """
    Function to weight the model.

    :param model_name: The model name to weight.
    :param data_path: Path to the directory containing the dataset.
    :param pca_dims: The number of PCA dimensions to use.
    :return: The weighted model.
    """
    # Load the trained model
    model = StaticModel.from_pretrained(model_name)

    logging.info("Applying reweighting and PCA to the model.")

    # Collect data for counting
    paths = sorted(Path(data_path).glob("*.json"))
    txt, _ = collect_means_and_texts(paths)

    counts: Counter[str] = Counter()
    for t in tqdm(txt):
        counts.update(model.tokenizer.encode(t, add_special_tokens=False).ids)

    sum_id = sum(counts.values()) + len(model.tokens)
    x = np.full(len(model.embedding), 1 / sum_id)

    # Weight the embeddings based on frequency
    for word_id, count in counts.items():
        x[word_id] = (count + 1) / sum_id

    w = model.embedding
    w = np.nan_to_num(w)

    # Apply PCA
    p = PCA(n_components=pca_dims)
    w = p.fit_transform(w)

    # Apply SIF weighting
    alpha = 1e-3
    f = alpha / (alpha + x)
    w *= f[:, None]
    model.embedding = w
    model.normalize = True

    model.save_pretrained(f"{model_name}_weighted")

    return model


def main(args: Any) -> None:
    """Main function."""
    train_model(
        args.model_name, args.data_path, args.save_path, device=args.device, random_embeddings=args.random_embeddings
    )
    weight_model(args.save_path, args.data_path, 256)


if __name__ == "__main__":
    # Define CLI arguments
    parser = argparse.ArgumentParser(description="Train a Model2Vec using tokenlearn.")

    # Training arguments
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

    main(args)
