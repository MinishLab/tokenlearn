import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
from model2vec import StaticModel
from model2vec.distill import distill
from sklearn.decomposition import PCA

from tokenlearn.pretrain import TextDataset, train_supervised
from tokenlearn.utils import collect_means_and_texts, create_vocab

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

_MAX_N_VAL_SAMPLES = 10_000


def train_model(
    model: StaticModel,
    train_txt: list[str],
    train_vec: np.ndarray,
    device: str = "cpu",
    pca_dims: int = 256,
) -> StaticModel:
    """
    Train a tokenlearn model.

    :param model: The static model to distill further.
    :param train_txt: List of texts to train on.
    :param train_vec: List of vectors to train on.
    :param device: Device to run the training on.
    :param pca_dims: Number of dimensions to reduce the target embeddings to using PCA.
    :return: The trained model.
    """
    pca_for_targets = PCA(n_components=pca_dims)
    train_vec = pca_for_targets.fit_transform(train_vec)
    var = np.cumsum(pca_for_targets.explained_variance_ratio_)[-1]
    logger.info(f"Explained variance of target embeddings: {var:.2f}")

    # Split the data into training and validation sets
    # We use a max of 10k samples as validation data
    val_samples = min(_MAX_N_VAL_SAMPLES, len(train_txt) // 10)
    train_txt, train_vec, val_txt, val_vec = (
        train_txt[:-val_samples],
        train_vec[:-val_samples],
        train_txt[-val_samples:],
        train_vec[-val_samples:],
    )

    train_data = TextDataset(train_txt, torch.from_numpy(train_vec), model.tokenizer)
    val_data = TextDataset(val_txt, torch.from_numpy(val_vec), model.tokenizer)

    # Train the model
    model = train_supervised(train_dataset=train_data, validation_dataset=val_data, model=model, device=device)
    return model


def main() -> None:
    """Main function to train and save a Model2Vec model using tokenlearn."""
    parser = argparse.ArgumentParser(description="Train a Model2Vec using tokenlearn.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="baai/bge-base-en-v1.5",
        help="The model name for distillation (e.g., 'baai/bge-base-en-v1.5').",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/fineweb_bgebase",
        help="Path to the directory containing the dataset.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the training on (e.g., 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=56000,
        help="The vocabulary size to use for training.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading the model.",
    )
    parser.add_argument(
        "--pca-dims",
        type=int,
        default=256,
        help="Number of dimensions to reduce the target embeddings to using PCA.",
    )
    args = parser.parse_args()

    # Collect paths for training data
    paths = sorted(Path(args.data_path).glob("*.json"))
    train_txt, train_vec = collect_means_and_texts(paths)

    pca_dims = args.pca_dims

    vocab_size = args.vocab_size
    if vocab_size:
        # Create a vocabulary if a vocab size is specified
        vocab = create_vocab(texts=train_txt, vocab_size=vocab_size)
        logger.info(f"Vocabulary created with {len(vocab)} tokens.")
    else:
        vocab = None
    model = distill(
        model_name=args.model_name, quantize_to="float32", vocabulary=vocab, pca_dims=pca_dims, trust_remote_code=True
    )

    # Train the model
    model = train_model(model, train_txt, train_vec, device=args.device, pca_dims=pca_dims)
    model.save_pretrained(args.save_path)


if __name__ == "__main__":
    main()
