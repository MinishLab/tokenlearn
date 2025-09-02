from enum import Enum
from typing import Callable

import torch
from torch import nn
from torch.nn.functional import cosine_similarity, cross_entropy


class Loss(str, Enum):
    """Different loss functions."""

    MSE = "mse"
    COSINE = "cosine"
    RANKING = "ranking"


def get_loss_function(loss: Loss) -> Callable:
    """Get the loss function."""
    if loss == Loss.MSE:
        return nn.MSELoss()
    elif loss == Loss.COSINE:
        return CosineSimilarityLoss()
    elif loss == Loss.RANKING:
        return PairwiseRankingLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss}")


class CosineSimilarityLoss(nn.Module):
    def __init__(self) -> None:
        """Initialize."""
        super().__init__()

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute the cosine similarity loss.

        :param X: Tensor of shape (batch_size, d)
        :param Y: Tensor of shape (batch_size, d)
        :return: Scalar loss
        """
        cos_sim = cosine_similarity(X, Y)
        loss = 1 - cos_sim.mean()
        return loss


class PairwiseRankingLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        """A simple pairwise ranking loss."""
        super().__init__()
        self.temperature = temperature

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute the pairwise ranking loss.

        :param X: Tensor of shape (batch_size, d)
        :param Y: Tensor of shape (batch_size, d)
        :return: Scalar loss
        """
        # Unsqueeze to get (B, B) matrix.
        dist = cosine_similarity(X.unsqueeze(1), Y.unsqueeze(0), dim=-1)
        logits = dist / self.temperature
        targets = torch.arange(len(dist), device=dist.device)
        return cross_entropy(logits, targets)
