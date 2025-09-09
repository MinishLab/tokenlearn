from __future__ import annotations

import logging
from tempfile import TemporaryDirectory
from typing import Any, Sequence

import pytorch_lightning as pl
import torch
from model2vec.train.base import FinetunableStaticModel, TextDataset
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from torch import nn
from torch.optim.adamw import AdamW

from tokenlearn.losses import Loss, get_loss_function

logger = logging.getLogger(__name__)

_RANDOM_SEED = 42
_MAX_VAL_SAMPLES = 10_000


class StaticModelForFineTuning(FinetunableStaticModel):
    def __init__(
        self, vectors: torch.Tensor, tokenizer: Tokenizer, out_dim: int, pad_id: int, loss: Loss | str, **kwargs: Any
    ) -> None:
        """
        Initialize from a model.

        :param vectors: The vectors to use.
        :param tokenizer: The tokenizer of the static model.
        :param out_dim: The output dimension. This should match the dimensionality of the targets.
        :param pad_id: The padding id. This can be any id.
        :param loss: The Enum of the loss function to use during training. This is not the function itself,
            but an enum which is used to later select the correct loss function.
        :param **kwargs: Additional keyword arguments passed to the parent class.
        """
        self.loss = Loss(loss)
        super().__init__(vectors=vectors, tokenizer=tokenizer, out_dim=out_dim, pad_id=pad_id, **kwargs)

    def construct_weights(self) -> nn.Parameter:
        """Set the weights."""
        weights = torch.ones(self.vectors.shape[0], dtype=torch.float32)
        weights[self.pad_id] = 0
        return nn.Parameter(weights)

    def fit(
        self,
        X: Sequence[str],
        y: torch.Tensor,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        min_epochs: int | None = None,
        max_epochs: int | None = None,
        early_stopping_patience: int | None = 5,
        test_size: float = 0.1,
        device: str = "auto",
        use_wandb: bool = False,
        project_name: str = "tokenlearn",
        run_name: str | None = None,
    ) -> StaticModelForFineTuning:
        """
        Fit a model.

        This function creates a Lightning Trainer object and fits the model to the data.
        It supports both single-label and multi-label classification.
        We use early stopping. After training, the weights of the best model are loaded back into the model.

        This function seeds everything with a seed of 42, so the results are reproducible.
        It also splits the data into a train and validation set, again with a random seed.

        :param X: The texts to train on.
        :param y: The labels to train on. If the first element is a list, multi-label classification is assumed.
        :param learning_rate: The learning rate.
        :param batch_size: The batch size.
        :param min_epochs: The minimum number of epochs to train for.
        :param max_epochs: The maximum number of epochs to train for.
            If this is -1, the model trains until early stopping is triggered.
        :param early_stopping_patience: The patience for early stopping.
            If this is None, early stopping is disabled.
        :param test_size: The test size for the train-test split.
        :param device: The device to train on. If this is "auto", the device is chosen automatically.
        :param use_wandb: Whether to use Weights & Biases for logging.
        :param project_name: The name of the Weights & Biases project. Ignored if `use_wandb` is False.
        :param run_name: The name of the Weights & Biases run. Ignored if `use_wandb` is False.
        :return: The fitted model.
        """
        pl.seed_everything(_RANDOM_SEED)
        # Determine whether the task is multilabel based on the type of y.
        test_size = min(_MAX_VAL_SAMPLES, int(len(X) * test_size))
        train_texts, validation_texts, train_labels, validation_labels = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )

        logger.info("Preparing training dataset.")
        train_dataset = self._prepare_dataset(train_texts, train_labels)
        logger.info("Preparing validation dataset.")
        val_dataset = self._prepare_dataset(validation_texts, validation_labels)

        c = _FineTuningLightningModule(self, learning_rate=learning_rate, loss_name=self.loss)

        n_train_batches = len(train_dataset) // batch_size
        callbacks: list[Callback] = []
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
        callbacks.append(checkpoint_callback)
        if early_stopping_patience is not None:
            callback = EarlyStopping(monitor="val_loss", mode="min", patience=early_stopping_patience)
            callbacks.append(callback)
        if use_wandb:
            wandb_logger = WandbLogger(project=project_name, name=run_name)
            lr_monitor = LearningRateMonitor(logging_interval="step")
            callbacks.append(lr_monitor)
        else:
            wandb_logger = None

        # If the dataset is small, we check the validation set every epoch.
        # If the dataset is large, we check the validation set every 250 batches.
        if n_train_batches < 250:
            val_check_interval = None
            check_val_every_epoch = 1
        else:
            val_check_interval = max(250, 2 * len(val_dataset) // batch_size)
            check_val_every_epoch = None

        with TemporaryDirectory() as tempdir:
            trainer = pl.Trainer(
                min_epochs=min_epochs,
                max_epochs=max_epochs,
                callbacks=callbacks,
                val_check_interval=val_check_interval,
                check_val_every_n_epoch=check_val_every_epoch,
                accelerator=device,
                default_root_dir=tempdir,
                logger=wandb_logger,
            )

            trainer.fit(
                c,
                train_dataloaders=train_dataset.to_dataloader(shuffle=True, batch_size=batch_size),
                val_dataloaders=val_dataset.to_dataloader(shuffle=False, batch_size=batch_size),
            )

            best_model_path = checkpoint_callback.best_model_path
            best_model_weights = torch.load(best_model_path, weights_only=True)

        state_dict = {}
        for weight_name, weight in best_model_weights["state_dict"].items():
            state_dict[weight_name.removeprefix("model.")] = weight

        self.load_state_dict(state_dict)
        self.eval()
        return self

    def _prepare_dataset(self, X: list[str], y: torch.Tensor, max_length: int = 512) -> TextDataset:
        """
        Prepare a dataset. For multilabel classification, each target is converted into a multi-hot vector.

        :param X: The texts.
        :param y: The labels.
        :param max_length: The maximum length of the input.
        :return: A TextDataset.
        """
        # This is a speed optimization.
        # assumes a mean token length of 10, which is really high, so safe.
        truncate_length = max_length * 10
        X = [x[:truncate_length] for x in X]
        tokenized: list[list[int]] = [
            encoding.ids[:max_length] for encoding in self.tokenizer.encode_batch_fast(X, add_special_tokens=False)
        ]
        return TextDataset(tokenized, y)


class _FineTuningLightningModule(pl.LightningModule):
    def __init__(self, model: StaticModelForFineTuning, learning_rate: float, loss_name: Loss) -> None:
        """Initialize the LightningModule."""
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_function = get_loss_function(loss_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simple forward pass."""
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step using cross-entropy loss for single-label and binary cross-entropy for multilabel training."""
        x, y = batch
        head_out, _ = self.model(x)
        loss = self.loss_function(head_out, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step computing loss and accuracy."""
        x, y = batch
        head_out, _ = self.model(x)
        loss = self.loss_function(head_out, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizer and learning rate scheduler."""
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        return optimizer
