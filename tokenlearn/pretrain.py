from __future__ import annotations

import logging

import numpy as np
import torch
from model2vec import StaticModel
from model2vec.distill.utils import select_optimal_device
from tokenizers import Tokenizer
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


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
        norms = vectors.norm(dim=1)
        # Normalize the vectors
        vectors = vectors / norms[:, None]
        self.embeddings = nn.Embedding.from_pretrained(vectors.clone(), freeze=False, padding_idx=pad_id)
        self.n_out = out_dim
        self.out_layer = nn.Linear(vectors.shape[1], self.n_out)
        weights = torch.Tensor(norms)
        weights[pad_id] = 0
        self.w = nn.Parameter(weights)

    def sub_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the mean."""
        w = self.w[input_ids]
        zeros = (input_ids != self.pad_id).float()
        w = w * zeros
        # Add a small epsilon to avoid division by zero
        length = zeros.sum(1) + 1e-16
        embedded = self.embeddings(input_ids)
        # Zero out the padding
        embedded = torch.bmm(w[:, None, :], embedded).squeeze(1)
        # Simulate actual mean
        embedded = embedded / length[:, None]

        return embedded

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the mean, and a classifier layer after."""
        embedded = self.sub_forward(x)
        return self.out_layer(embedded), embedded

    @property
    def device(self) -> torch.device:
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
        self.texts = [x[:20_000] for x in texts]
        self.tokenized_texts: list[list[int]] = [
            encoding.ids[:512] for encoding in tokenizer.encode_batch_fast(self.texts, add_special_tokens=False)
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
    validation_dataset: TextDataset,
    model: StaticModel,
    patience: int | None = 5,
    device: str | None = None,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> StaticModel:
    """
    Train a tokenlearn model.

    :param train_dataset: The training dataset.
    :param validation_dataset: The validation dataset.
    :param model: The model to train.
    :param patience: The number of epochs to wait before early stopping.
    :param device: The device to train on.
    :param batch_size: The batch size.
    :param lr: The learning rate.
    :return: The trained model.
    """
    device = select_optimal_device(device)
    train_dataloader = train_dataset.to_dataloader(shuffle=True, batch_size=batch_size)

    # Initialize the model
    pad_id = model.tokenizer.token_to_id("[PAD]") or model.tokenizer.token_to_id("<pad>")
    trainable_model = StaticModelFineTuner(
        torch.from_numpy(model.embedding),
        out_dim=train_dataset.targets.shape[1],
        pad_id=pad_id,
    )
    trainable_model.to(device)

    # Separate parameters for model and linear layer
    model_params = (
        list(trainable_model.embeddings.parameters())
        + [trainable_model.w]
        + list(trainable_model.out_layer.parameters())
    )

    # Create optimizer with separate parameter groups
    optimizer = torch.optim.AdamW(params=model_params, lr=lr)

    lowest_loss = float("inf")
    param_dict = trainable_model.state_dict()
    curr_patience = patience
    stop = False

    criterion = nn.MSELoss()

    try:
        for epoch in range(100_000):
            logger.info(f"Epoch {epoch}")
            trainable_model.train()

            # Track train loss separately
            train_losses = []
            barred_train = tqdm(train_dataloader, desc=f"Epoch {epoch:03d} [Train]")

            for idx, (x, y) in enumerate(barred_train):
                optimizer.zero_grad()
                x = x.to(trainable_model.device)
                y_hat, _ = trainable_model(x)
                # Separate loss components
                train_loss = criterion(y_hat, y.to(trainable_model.device)).mean()

                # Apply weights
                train_loss.backward()

                optimizer.step()
                train_losses.append(train_loss.item())

                barred_train.set_description_str(f"Train Loss: {np.mean(train_losses[-10:]):.3f}")

                # Evaluate every 1000 steps and at the end of the epoch
                if (idx > 0 and idx % 1000 == 0) or idx == len(train_dataloader) - 1:
                    trainable_model.eval()
                    with torch.no_grad():
                        validation_losses = []
                        barred_val = tqdm(
                            validation_dataset.to_dataloader(shuffle=False, batch_size=batch_size), desc="Validation"
                        )
                        for x_val, y_val in barred_val:
                            x_val = x_val.to(trainable_model.device)
                            y_hat_val, _ = trainable_model(x_val)
                            val_loss = criterion(y_hat_val, y_val.to(trainable_model.device)).mean()
                            validation_losses.append(val_loss.item())
                            barred_val.set_description_str(f"Validation Loss: {np.mean(validation_losses):.3f}")

                        validation_loss = np.mean(validation_losses)
                    # Early stopping logic based on validation loss
                    if patience is not None and curr_patience is not None:
                        if (lowest_loss - validation_loss) > 1e-4:
                            param_dict = trainable_model.state_dict()  # Save best model state based on training loss
                            curr_patience = patience
                            lowest_loss = validation_loss
                        else:
                            curr_patience -= 1
                            if curr_patience == 0:
                                stop = True
                                break
                        logger.info(f"Patience level: {patience - curr_patience}")
                        logger.info(f"Validation loss: {validation_loss:.3f}")
                        logger.info(f"Lowest loss: {lowest_loss:.3f}")

                    trainable_model.train()

            if stop:
                logger.info("Early stopping")
                break

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

    return new_model
