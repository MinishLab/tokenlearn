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
from tokenlearn.utils import calculate_token_probabilities, collect_means_and_texts, create_vocab

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def train_model(
    model_name: str, train_txt: list[str], train_vec: np.ndarray, device: str = "cpu", vocab_size: int | None = None
) -> StaticModel:
    """
    Train a tokenlearn model.

    :param model_name: The sentence transformer model name for distillation.
    :param train_txt: List of texts to train on.
    :param train_vec: List of vectors to train on.
    :param device: Device to run the training on.
    :param vocab_size: The vocabulary size to use (optional).
    :return: The trained model.
    """
    if vocab_size:
        # Create a vocabulary if a vocab size is specified
        vocab = create_vocab(texts=train_txt, vocab_size=vocab_size)
        model = distill(model_name=model_name, vocabulary=vocab)
        logger.info(f"Vocabulary size: {len(vocab)}")
    else:
        model = distill(model_name=model_name)
    train_data = TextDataset(train_txt, torch.from_numpy(train_vec), model.tokenizer)

    # Train the model
    model, _ = train_supervised(train_dataset=train_data, model=model, device=device)
    return model


def weight_model(model: StaticModel, text: List[str], pca_dims: int, alpha: float = 1e-3) -> StaticModel:
    """
    Function to weight the model.

    :param model: The model to weight.
    :param text: The text to use for weighting.
    :param pca_dims: The number of PCA dimensions to use.
    :param alpha: The alpha value for SIF weighting. Words with probabilities above this value will be downweighted.
    :return: The weighted model.
    """
    logging.info("Applying reweighting and PCA to the model.")
    probas = calculate_token_probabilities(model.tokenizer, text)

    w = model.embedding
    w = np.nan_to_num(w)

    # Apply PCA
    p = PCA(n_components=pca_dims)
    w = p.fit_transform(w)

    # Apply SIF weighting
    f = alpha / (alpha + probas)
    w *= f[:, None]
    model.embedding = w
    model.normalize = True

    return model


def save_model(model: StaticModel, save_path: str, is_weighted: bool = False) -> None:
    """
    Save the model to the specified path.

    :param model: The model to save.
    :param save_path: Path to save the model.
    :param is_weighted: Whether the model is weighted.
    """
    if is_weighted:
        save_path = f"{save_path}_weighted"
    model.save_pretrained(save_path)
    logging.info(f"Model saved to {save_path}")


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
        default=None,
        help="The vocabulary size to use for training.",
    )
    args = parser.parse_args()

    # Collect paths for training data
    paths = sorted(Path(args.data_path).glob("*.json"))
    train_txt, train_vec = collect_means_and_texts(paths)

    # Train the model
    model = train_model(args.model_name, train_txt, train_vec, device=args.device, vocab_size=args.vocab_size)
    save_model(model, args.save_path)

    # Apply weighting and save the weighted model
    weighted_model = weight_model(model, train_txt, pca_dims=256)
    save_model(weighted_model, args.save_path, is_weighted=True)


if __name__ == "__main__":
    main()
