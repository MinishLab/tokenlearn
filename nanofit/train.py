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


def train_supervised(
    train_dataset: TextDataset,
    val_dataset: TextDataset,
    model: StaticModel,
    max_epochs: int = 100,
    lr: float = 1e-3,
    patience: int | None = 3,
    device: str = "cpu",
    batch_size: int = 256,
) -> StaticModel:
    """Train a supervised classifier using cross entropy."""
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
            barred_train = tqdm(train_dataloader, desc="{:03d}".format(epoch))
            loss_avg = 0.0
            n = 0
            losses = []
            for x, y in barred_train:
                # Backward pass of the model and optimizer step.
                optimizer.zero_grad()
                x = x.to(trainable_model.device)
                y_hat, emb = trainable_model(x)
                loss: torch.FloatTensor = 1 - criterion(y_hat, y.to(trainable_model.device)).mean()
                loss = loss + (emb**2).mean()
                loss.backward()
                optimizer.step()
                loss_avg += loss.item()
                n += 1

                losses.append(loss.item())
                losses = losses[-10:]

                barred_train.set_description_str(f"Loss: {np.mean(losses):.3f}")

            trainable_model.eval()

            with torch.no_grad():
                loss_avg = 0.0
                n = 0
                for x, y in val_dataloader:
                    x = x.to(trainable_model.device)
                    y_hat, emb = trainable_model(x)
                    loss = 1 - criterion(y_hat, y.to(trainable_model.device)).mean()
                    loss = loss + (emb**2).mean()
                    loss_avg += loss.item()
                    n += 1

                loss_avg /= n

                if patience is not None and curr_patience is not None:
                    if (lowest_loss - loss_avg) > _MIN_DELTA:
                        param_dict = trainable_model.state_dict()
                        curr_patience = patience
                        lowest_loss = loss_avg
                    else:
                        curr_patience -= 1
                        if curr_patience == 0:
                            break
                    patience_str = "🌝" * curr_patience
                    logger.info(f"Patience level: {patience_str}")
                    logger.info(f"Lowest loss: {lowest_loss:.3f}")

                logger.info(f"Validation loss: {loss_avg:.3f}")

            trainable_model.train()
    except KeyboardInterrupt:
        logger.info("Interruptor")
    trainable_model.eval()
    # Load best model
    trainable_model.load_state_dict(param_dict)

    with torch.no_grad():
        vectors = (
            trainable_model.sub_forward(torch.arange(len(trainable_model.embeddings.weight))[:, None]).cpu().numpy()
        )

    new_model = StaticModel(vectors=vectors, tokenizer=model.tokenizer, config=model.config)

    return new_model, trainable_model
