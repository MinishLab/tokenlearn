import logging
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from model2vec import StaticModel
from model2vec.distill import distill
from model2vec.train import StaticModelForSimilarity
from sklearn.decomposition import PCA

from tokenlearn.utils import collect_means_and_texts, create_vocab

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


_DEFAULT_BATCH_SIZE = 256
_DEFAULT_LEARNING_RATE = 1e-3


def _parse_args() -> Namespace:
    parser = ArgumentParser(description="Train a Model2Vec using tokenlearn.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model-name",
        type=str,
        help="The model name for distillation (e.g., 'baai/bge-base-en-v1.5').",
    )
    group.add_argument(
        "--model2vec-model-name",
        type=str,
        help="The Model2Vec model name to initialize from (e.g., 'vocquant').",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="minishlab/tokenlearn-c4-en-bge-base-en-v1.5",
        help="Path to a local HuggingFace dataset directory or a Hub repo ID.",
    )
    parser.add_argument(
        "--data-split",
        type=str,
        default="train",
        help="Dataset split to use when loading from the Hub (e.g., 'train', 'validation').",
    )
    parser.add_argument(
        "--data-name",
        type=str,
        default=None,
        help="Dataset configuration name when loading from the Hub (e.g., 'en' for C4).",
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
        help="Number of dimensions for the model2vec PCA.",
    )
    parser.add_argument(
        "--target-pca-dims",
        type=int,
        help="Number of dimensions to reduce the target embeddings to using PCA. If not provided, PCA is not applied",
    )
    parser.add_argument("--limit-samples", type=int, help="Limit the number of samples to use for training.")
    parser.add_argument("--lr", default=_DEFAULT_LEARNING_RATE, type=float, help="Learning rate for training.")
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE, help="Batch size for training.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # Collect paths for training data
    train_txt, train_vec = collect_means_and_texts(
        args.data_path, args.limit_samples, split=args.data_split, name=args.data_name
    )

    pca_dims = args.pca_dims

    if args.model2vec_model_name:
        if args.vocab_size:
            logger.warning("Ignoring --vocab-size since --model2vec-model-name is provided.")
        model = StaticModel.from_pretrained(
            path=args.model2vec_model_name,
            quantize_to="float32",
        )
    else:
        vocab: list[str] | None = None
        vocab_size = args.vocab_size
        if vocab_size:
            # Create a vocabulary if a vocab size is specified
            vocab = create_vocab(texts=train_txt, vocab_size=vocab_size)
            logger.info(f"Vocabulary created with {len(vocab)} tokens.")
        model = distill(
            model_name=args.model_name,
            quantize_to="float32",
            vocabulary=vocab,
            pca_dims=pca_dims,
            trust_remote_code=True,
        )

    # Train the model
    target_pca_dims = args.target_pca_dims
    if target_pca_dims:
        pca_for_targets = PCA(n_components=target_pca_dims)
        train_vec = pca_for_targets.fit_transform(train_vec)
        var = np.cumsum(pca_for_targets.explained_variance_ratio_)[-1]
        logger.info(f"Explained variance of target embeddings: {var:.2f}")

    trainable = StaticModelForSimilarity.from_static_model(
        model=model,
        out_dim=train_vec.shape[1],
        n_layers=0,
        freeze_weights=True,
    )
    trainable.fit(
        X=train_txt,
        y=torch.from_numpy(train_vec),
        batch_size=args.batch_size,
        device=args.device,
        learning_rate=args.lr,
    )

    trainable.to_static_model().save_pretrained(args.save_path)
