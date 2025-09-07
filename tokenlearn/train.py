import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from model2vec.distill import distill
from sklearn.decomposition import PCA

from tokenlearn.losses import Loss
from tokenlearn.model import StaticModelForFineTuning
from tokenlearn.utils import collect_means_and_texts, create_vocab

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


_DEFAULT_BATCH_SIZE = 256
_DEFAULT_LEARNING_RATE = 1e-3


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
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases for logging.")
    parser.add_argument("--project-name", type=str, default="tokenlearn", help="Weights & Biases project name.")
    parser.add_argument("--run-name", type=str, help="Weights & Biases run name.")
    parser.add_argument(
        "--limit-samples", type=int, default=None, help="Limit the number of samples to use for training."
    )
    parser.add_argument(
        "--loss",
        default="contrastive",
        choices=Loss.__members__.values(),
        help="The loss function to use for training.",
    )
    parser.add_argument("--lr", default=_DEFAULT_LEARNING_RATE, type=float, help="Learning rate for training.")
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE, help="Batch size for training.")
    args = parser.parse_args()

    # Collect paths for training data
    paths = sorted(Path(args.data_path).glob("*.json"))
    train_txt, train_vec = collect_means_and_texts(paths, args.limit_samples)

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
    pca_for_targets = PCA(n_components=pca_dims)
    train_vec = pca_for_targets.fit_transform(train_vec)
    var = np.cumsum(pca_for_targets.explained_variance_ratio_)[-1]
    logger.info(f"Explained variance of target embeddings: {var:.2f}")

    model_name = args.model_name.split("/")[-1]
    dataset_name = args.data_path.split("/")[-1]
    limit = args.limit_samples

    if not args.run_name:
        run_name = f"{model_name}-{dataset_name}"
        if limit is not None:
            run_name = f"{run_name}-limit{limit}"
    else:
        run_name = args.run_name

    loss = Loss(args.loss)

    trainable = StaticModelForFineTuning.from_static_model(model=model, out_dim=train_vec.shape[1], loss=loss)
    trainable.fit(
        X=train_txt,
        y=torch.from_numpy(train_vec),
        batch_size=args.batch_size,
        device=args.device,
        use_wandb=args.use_wandb,
        project_name=args.project_name,
        run_name=run_name,
        learning_rate=args.lr,
    )

    trainable.to_static_model().save_pretrained(args.save_path)


if __name__ == "__main__":
    main()
