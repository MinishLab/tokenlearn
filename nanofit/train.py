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

_MIN_DELTA = 0.001

# Try to import wandb, if not available, set a flag
try:
    import wandb

    wandb_installed = True
except ImportError:
    logger.warning("wandb is not installed. Skipping wandb logging.")
    wandb_installed = False


def init_wandb(project_name: str = "minishlab", config: dict | None = None) -> None:
    """Initialize Weights & Biases for tracking experiments if wandb is installed."""
    if wandb_installed:
        wandb.init(project=project_name, config=config)
        logger.info(f"W&B initialized with project: {project_name}")
    else:
        logger.info("Skipping W&B initialization since wandb is not installed.")


class StaticModelFineTuner(nn.Module):
    def __init__(self, vectors: torch.Tensor, out_dim: int, pad_id: int) -> None:
        """Initialize from a model."""
        super().__init__()
        self.embeddings = nn.Embedding.from_pretrained(vectors.clone(), freeze=False, padding_idx=0)
        self.n_out = out_dim
        self.out_layer = nn.Linear(vectors.shape[1], self.n_out)
        weights = torch.ones(len(vectors))
        weights[pad_id] = 0
        self.w = nn.Parameter(weights)

    def sub_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the mean."""
        w = self.w[x]
        zeros = (x != 0).float()
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
        """Initialize the dataset."""
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
        """Gets an item :)))))."""
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
    val_dataset: TextDataset,
    model: StaticModel,
    max_epochs: int = 200,
    min_epochs: int = 5,
    lr: float = 1e-3,
    patience: int | None = 3,
    device: str = "cpu",
    batch_size: int = 256,
    project_name: str = "minishlab",
    config: dict | None = None,
) -> StaticModel:
    """Train a supervised classifier using cross-entropy loss and track metrics with W&B if available."""
    if config is None:
        config = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "min_epochs": min_epochs,
            "patience": patience,
        }

    # Initialize W&B only if wandb is installed
    if wandb_installed:
        init_wandb(project_name=project_name, config=config)

    train_dataloader = train_dataset.to_dataloader(shuffle=True, batch_size=batch_size)
    val_dataloader = val_dataset.to_dataloader(shuffle=False, batch_size=batch_size)

    trainable_model = StaticModelFineTuner(
        torch.from_numpy(model.embedding),
        out_dim=train_dataset.targets.shape[1],
        pad_id=model.tokenizer.token_to_id("[PAD]"),
    )
    trainable_model.to(device)
    optimizer = torch.optim.Adam(trainable_model.parameters(), lr=lr)
    criterion = nn.CosineSimilarity()

    lowest_loss = float("inf")
    param_dict = trainable_model.state_dict()
    curr_patience = patience

    try:
        for epoch in range(max_epochs):
            logger.info(f"Epoch {epoch}")
            trainable_model.train()

            # Track train loss separately
            train_losses = []
            barred_train = tqdm(train_dataloader, desc=f"Epoch {epoch:03d} [Train]")

            for x, y in barred_train:
                optimizer.zero_grad()
                x = x.to(trainable_model.device)
                y_hat, emb = trainable_model(x)
                train_loss = 1 - criterion(y_hat, y.to(trainable_model.device)).mean()
                train_loss = train_loss + (emb**2).mean()
                train_loss.backward()
                optimizer.step()

                train_losses.append(train_loss.item())
                barred_train.set_description_str(f"Train Loss: {np.mean(train_losses):.3f}")

            # Calculate average train loss
            avg_train_loss = np.mean(train_losses)

            # Validation phase
            trainable_model.eval()
            val_losses = []
            with torch.no_grad():
                barred_val = tqdm(val_dataloader, desc=f"Epoch {epoch:03d} [Val]")
                for x, y in barred_val:
                    x = x.to(trainable_model.device)
                    y_hat, emb = trainable_model(x)
                    val_loss = 1 - criterion(y_hat, y.to(trainable_model.device)).mean()
                    val_loss = val_loss + (emb**2).mean()
                    val_losses.append(val_loss.item())
                    barred_val.set_description_str(f"Val Loss: {np.mean(val_losses):.3f}")

            # Calculate average validation loss
            avg_val_loss = np.mean(val_losses)

            # Log both training and validation loss together, only if wandb is installed
            if wandb_installed:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,  # Log validation loss in the same step
                    }
                )

            # Early stopping logic
            if patience is not None and curr_patience is not None and epoch >= min_epochs:
                if (lowest_loss - avg_val_loss) > _MIN_DELTA:
                    param_dict = trainable_model.state_dict()
                    curr_patience = patience
                    lowest_loss = avg_val_loss
                else:
                    curr_patience -= 1
                    if curr_patience == 0:
                        break
                patience_str = "🌝" * curr_patience
                logger.info(f"Patience level: {patience_str}")
                logger.info(f"Lowest loss: {lowest_loss:.3f}")

            logger.info(f"Validation loss: {avg_val_loss:.3f}")
            trainable_model.train()

    except KeyboardInterrupt:
        logger.info("Training interrupted")

    trainable_model.eval()
    # Load best model
    trainable_model.load_state_dict(param_dict)

    # Move the embeddings to the device (GPU)
    embeddings_weight = trainable_model.embeddings.weight.to(device)

    # Perform the forward pass on GPU
    with torch.no_grad():
        vectors = trainable_model.sub_forward(torch.arange(len(embeddings_weight))[:, None].to(device)).cpu().numpy()

    new_model = StaticModel(vectors=vectors, tokenizer=model.tokenizer, config=model.config)

    return new_model, trainable_model
