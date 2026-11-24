"""Baseline multi-output MLP for per-patch NFE pass probabilities.

Outputs independent logits for each supported budget. No monotonicity is
assumed: a higher NFE can sometimes receive a lower predicted pass score.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PatchNFEPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        num_budgets: int = 9,
        hidden_dims: tuple[int, int] = (64, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim < 1 or num_budgets < 1 or len(hidden_dims) != 2:
            raise ValueError("Expected positive dimensions and two hidden layers")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], num_budgets),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return unnormalized logits [..., num_budgets], NOT class IDs."""
        return self.network(features)

    @torch.no_grad()
    def probabilities(self, features: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            return self(features).sigmoid()
        finally:
            self.train(was_training)


def masked_bce_loss(
    logits: torch.Tensor, targets: torch.Tensor, observed_mask: torch.Tensor,
) -> torch.Tensor:
    """Average BCE only over *observed* budget outcomes.

    Unknown labels must have mask=0. An observed failure is target=0, mask=1.
    Soft empirical pass fractions in [0, 1] are also supported.
    """
    if logits.shape != targets.shape or logits.shape != observed_mask.shape:
        raise ValueError("logits, targets and mask must have the same shape")
    if not torch.all((targets >= 0) & (targets <= 1)):
        raise ValueError("targets must be in [0, 1]")
    if not torch.all((observed_mask == 0) | (observed_mask == 1)):
        raise ValueError("mask must contain only 0 and 1")
    count = observed_mask.sum()
    if count.item() == 0:
        raise ValueError("No observed labels in batch")
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (losses * observed_mask).sum() / count


def choose_budget(probabilities: torch.Tensor, budgets: tuple[int, ...],
                  threshold: float, fallback: int) -> int:
    """Conservative selection helper. Threshold requires external calibration."""
    if probabilities.ndim != 1 or len(probabilities) != len(budgets):
        raise ValueError("Expected a 1D probability vector, one entry per budget")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0,1]")
    if fallback not in budgets or sorted(set(budgets)) != list(budgets):
        raise ValueError("budgets must be unique sorted values including fallback")
    for budget, prob in zip(budgets, probabilities.tolist()):
        if float(prob) >= threshold:
            return budget
    return fallback
