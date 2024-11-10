from __future__ import annotations

import logging

import numpy as np
import torch
from model2vec import StaticModel
from tokenizers import Tokenizer
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Try to import wandb for logging
try:
    import wandb

    _WANDB_INSTALLED = True
except ImportError:
    _WANDB_INSTALLED = False


def init_wandb(project_name: str, config: dict | None = None) -> None:
    """Initialize Weights & Biases for tracking experiments if wandb is installed."""
    if _WANDB_INSTALLED:
        wandb.init(project=project_name, config=config)
        logger.info(f"W&B initialized with project: {project_name}")
    else:
        logger.info("Skipping W&B initialization since wandb is not installed.")


class StaticModelFineTuner(nn.Module):
    def __init__(self, vectors: torch.Tensor, out_dim: int, pad_id: int) -> None:
        """
        Initialize from a model.

        :param vectors: The vectors to use.
        :param out_dim: The output dimension.
        :param pad_id: The padding id.
        """
        super().__init__()
        self.pad_id = pad_id
        self.embeddings = nn.Embedding.from_pretrained(vectors.clone(), freeze=False, padding_idx=pad_id)
        self.n_out = out_dim
        self.out_layer = nn.Linear(vectors.shape[1], self.n_out)
        weights = torch.ones(len(vectors))
        weights[pad_id] = 0
        self.w = nn.Parameter(weights)

    def sub_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the mean."""
        w = self.w[x]
        zeros = (x != self.pad_id).float()
        length = zeros.sum(1)
        embedded = self.embeddings(x)
        # Simulate actual mean
        # Zero out the padding
        embedded = embedded * zeros[:, :, None]
        embedded = (embedded * w[:, :, None]).sum(1) / w.sum(1)[:, None]
        embedded = embedded / length[:, None]

        return embedded

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the mean, and a classifier layer after."""
        embedded = self.sub_forward(x)
        return self.out_layer(embedded), embedded

    @property
    def device(self) -> str:
        """Get the device of the model."""
        return self.embeddings.weight.device


class TextDataset(Dataset):
    def __init__(self, texts: list[str], targets: torch.Tensor, tokenizer: Tokenizer) -> None:
        """
        Initialize the dataset.

        :param texts: The texts to tokenize.
        :param targets: The targets.
        :param tokenizer: The tokenizer to use.
        :raises ValueError: If the number of labels does not match the number of texts.
        """
        if len(targets) != len(texts):
            raise ValueError("Number of labels does not match number of texts.")
        self.texts = texts
        self.tokenized_texts: list[list[int]] = [
            encoding.ids for encoding in tokenizer.encode_batch_fast(self.texts, add_special_tokens=False)
        ]
        self.targets = targets
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.tokenized_texts)

    def __getitem__(self, index: int) -> tuple[list[int], torch.Tensor]:
        """Gets an item."""
        return self.tokenized_texts[index], self.targets[index]

    @staticmethod
    def collate_fn(batch: list[tuple[list[list[int]], int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate function."""
        texts, targets = zip(*batch)

        tensors = [torch.LongTensor(x).int() for x in texts]
        padded = pad_sequence(tensors, batch_first=True, padding_value=0)

        return padded, torch.stack(targets)

    def to_dataloader(self, shuffle: bool, batch_size: int = 32) -> DataLoader:
        """Convert the dataset to a DataLoader."""
        return DataLoader(self, collate_fn=self.collate_fn, shuffle=shuffle, batch_size=batch_size)


def train_supervised(  # noqa: C901
    train_dataset: TextDataset,
    model: StaticModel,
    max_epochs: int = 50,
    min_epochs: int = 1,
    patience: int | None = 5,
    patience_min_delta: float = 0.001,
    device: str = "cpu",
    batch_size: int = 256,
    project_name: str | None = None,
    config: dict | None = None,
    lr_scheduler_patience: int = 3,
    lr_scheduler_min_delta: float = 0.03,
    cosine_weight: float = 1.0,
    mse_weight: float = 1.0,
    lr_model: float = 0.003,
    lr_linear: float = 0.01,
) -> StaticModel:
    """
    Train a tokenlearn model.

    :param train_dataset: The training dataset.
    :param model: The model to train.
    :param max_epochs: The maximum number of epochs to train.
    :param min_epochs: The minimum number of epochs to train.
    :param patience: The number of epochs to wait before early stopping.
    :param patience_min_delta: The minimum delta for early stopping.
    :param device: The device to train on.
    :param batch_size: The batch size.
    :param project_name: The name of the project for W&B.
    :param config: The configuration for W&B.
    :param lr_scheduler_patience: The patience for the learning rate scheduler.
    :param lr_scheduler_min_delta: The minimum delta for the learning rate scheduler.
    :param cosine_weight: The weight for the cosine loss.
    :param mse_weight: The weight for the MSE loss.
    :param lr_model: The learning rate for the model.
    :param lr_linear: The learning rate for the linear layer.
    :return: The trained model.
    """
    if config is None:
        config = {
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "min_epochs": min_epochs,
            "patience": patience,
            "lr_scheduler_patience": lr_scheduler_patience,
            "lr_scheduler_min_delta": lr_scheduler_min_delta,
            "cosine_weight": cosine_weight,
            "mse_weight": mse_weight,
            "lr_model": lr_model,
            "lr_linear": lr_linear,
        }

    # Initialize W&B only if wandb is installed and project name is provided
    if _WANDB_INSTALLED and project_name:
        init_wandb(project_name=project_name, config=config)
        wandb_initialized = True
    else:
        wandb_initialized = False

    train_dataloader = train_dataset.to_dataloader(shuffle=True, batch_size=batch_size)

    # Initialize the model
    trainable_model = StaticModelFineTuner(
        torch.from_numpy(model.embedding),
        out_dim=train_dataset.targets.shape[1],
        pad_id=model.tokenizer.token_to_id("[PAD]"),
    )
    trainable_model.to(device)

    # Separate parameters for model and linear layer
    model_params = list(trainable_model.embeddings.parameters()) + [trainable_model.w]
    linear_params = trainable_model.out_layer.parameters()

    # Create optimizer with separate parameter groups
    optimizer = torch.optim.Adam([{"params": model_params, "lr": lr_model}, {"params": linear_params, "lr": lr_linear}])

    criterion = nn.CosineSimilarity()

    # Initialize the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=lr_scheduler_patience,
        verbose=True,
        min_lr=1e-6,
        threshold=lr_scheduler_min_delta,
        threshold_mode="rel",
    )

    lowest_loss = float("inf")
    param_dict = trainable_model.state_dict()
    curr_patience = patience

    try:
        for epoch in range(max_epochs):
            logger.info(f"Epoch {epoch}")
            trainable_model.train()

            # Track train loss separately
            train_losses = []
            cosine_losses = []
            mse_losses = []
            barred_train = tqdm(train_dataloader, desc=f"Epoch {epoch:03d} [Train]")

            for x, y in barred_train:
                optimizer.zero_grad()
                x = x.to(trainable_model.device)
                y_hat, emb = trainable_model(x)
                # Separate loss components
                cosine_loss = 1 - criterion(y_hat, y.to(trainable_model.device)).mean()
                mse_loss = (emb**2).mean()

                # Apply weights
                total_loss = cosine_weight * cosine_loss + mse_weight * mse_loss
                total_loss.backward()

                optimizer.step()

                train_losses.append(total_loss.item())
                cosine_losses.append(cosine_loss.item())
                mse_losses.append(mse_loss.item())

                barred_train.set_description_str(f"Train Loss: {np.mean(train_losses):.3f}")

            # Calculate average losses
            avg_train_loss = np.mean(train_losses)
            avg_cosine_loss = np.mean(cosine_losses)
            avg_mse_loss = np.mean(mse_losses)

            # Step the scheduler with the current training loss
            scheduler.step(avg_train_loss)

            # Get current learning rates
            current_lr_model = optimizer.param_groups[0]["lr"]
            current_lr_linear = optimizer.param_groups[1]["lr"]

            # Log training loss and learning rates to wandb
            if wandb_initialized:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": avg_train_loss,
                        "cosine_loss": avg_cosine_loss,
                        "mse_loss": avg_mse_loss,
                        "learning_rate_model": current_lr_model,
                        "learning_rate_linear": current_lr_linear,
                    }
                )

            # Early stopping logic based on training loss
            if patience is not None and curr_patience is not None and epoch >= min_epochs:
                if (lowest_loss - avg_train_loss) > patience_min_delta:
                    param_dict = trainable_model.state_dict()  # Save best model state based on training loss
                    curr_patience = patience
                    lowest_loss = avg_train_loss
                else:
                    curr_patience -= 1
                    if curr_patience == 0:
                        break
                patience_str = "🌝" * curr_patience
                logger.info(f"Patience level: {patience_str}")
                logger.info(f"Lowest train loss: {lowest_loss:.3f}")

            logger.info(f"Training loss: {avg_train_loss:.3f}")
            logger.info(f"Cosine loss: {avg_cosine_loss:.3f}")
            logger.info(f"MSE loss: {avg_mse_loss:.3f}")
            trainable_model.train()

    except KeyboardInterrupt:
        logger.info("Training interrupted")

    trainable_model.eval()
    # Load best model based on training loss
    trainable_model.load_state_dict(param_dict)

    # Move the embeddings to the device (GPU)
    embeddings_weight = trainable_model.embeddings.weight.to(device)

    # Perform the forward pass on GPU
    with torch.no_grad():
        vectors = trainable_model.sub_forward(torch.arange(len(embeddings_weight))[:, None].to(device)).cpu().numpy()

    new_model = StaticModel(vectors=vectors, tokenizer=model.tokenizer, config=model.config)

    return new_model, trainable_model
