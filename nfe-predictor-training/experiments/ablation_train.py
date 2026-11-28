"""Train and score one feature matrix with the baseline model, loss and metrics.

training.train.fit is tied to the six canonical inputs, so this mirrors its loop
(seeding, Adam, masked BCE, best validation epoch, test untouched) for any input
width, and reuses the team's model, loss, budget choice and metric functions.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import TensorDataset

from models.mlp import PatchNFEPredictor, masked_bce_loss
from experiments.sequence_cost import CostModel, DEFAULT_POLICIES, select_and_score
from training.evaluate import compare_fixed_nfe, per_budget_metrics
from training.train import TrainingConfig, make_loader, seed_everything

DEFAULT_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)


@dataclass(frozen=True)
class SplitArrays:
    x: np.ndarray  # [n, d], already standardized
    y: np.ndarray  # [n, budgets]
    m: np.ndarray  # [n, budgets]
    sequence_ids: tuple[str, ...] = ()  # request of each row, for switch-aware cost
    positions: tuple[int, ...] = ()     # patch_index of each row

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (torch.tensor(self.x, dtype=torch.float32),
                torch.tensor(self.y, dtype=torch.float32),
                torch.tensor(self.m, dtype=torch.float32))


def fit_standardizer(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Population mean/std on training rows; constant columns get scale 1."""
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    return mean, np.where(std == 0, 1.0, std)


def _mean_loss(model, arrays: SplitArrays) -> float:
    x, y, m = arrays.tensors()
    with torch.no_grad():
        return float(masked_bce_loss(model(x), y, m))


def train_mlp(train: SplitArrays, val: SplitArrays, config: TrainingConfig) -> PatchNFEPredictor:
    """Masked-BCE fit, keeping the epoch with the lowest validation loss."""
    seed_everything(config.seed)
    model = PatchNFEPredictor(input_dim=train.x.shape[1], num_budgets=train.y.shape[1])
    loader = make_loader(TensorDataset(*train.tensors()), batch_size=config.batch_size,
                         shuffle=True, seed=config.seed, num_workers=config.num_workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    best_loss, best_state = float("inf"), None
    for _ in range(config.epochs):
        model.train()
        for x, y, m in loader:
            if not m.sum():
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = masked_bce_loss(model(x), y, m)
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = _mean_loss(model, val)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model


def probabilities(model, arrays: SplitArrays) -> torch.Tensor:
    with torch.no_grad():
        return model(arrays.tensors()[0]).sigmoid()


def brier(p: torch.Tensor, arrays: SplitArrays, budgets: tuple[int, ...]) -> float:
    """Brier score over every observed (example, budget) outcome."""
    per = per_budget_metrics(p, *arrays.tensors()[1:], budgets)
    observed = sum(v["observed"] for v in per.values())
    return sum(v["brier"] * v["observed"] for v in per.values() if v["observed"]) / observed


def choose_threshold(
    p_val: torch.Tensor, val: SplitArrays, budgets: tuple[int, ...], *, fixed_nfe: int,
    fallback_nfe: int, thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> float | None:
    """On validation: the lowest-mean-NFE threshold whose pass rate is at least fixed NFE's.

    Ties go to the higher (more conservative) threshold. None if no threshold matches
    the fixed budget's quality or there are no paired outcomes.
    """
    _, y, m = val.tensors()
    best = None
    for t in sorted(thresholds):
        r = compare_fixed_nfe(p_val, y, m, fixed_nfe=fixed_nfe, threshold=t,
                              fallback_nfe=fallback_nfe, budgets=budgets)
        if not r["paired_observed"] or r["adaptive_pass_rate"] < r["fixed_pass_rate"]:
            continue
        if best is None or r["adaptive_mean_nfe"] <= best[1]:
            best = (t, r["adaptive_mean_nfe"])
    return None if best is None else best[0]


def oracle_mean_nfe(arrays: SplitArrays, budgets: tuple[int, ...], fallback_nfe: int) -> float:
    """Mean of the smallest observed passing budget (fallback if none): an optimistic floor."""
    chosen = []
    for y, m in zip(arrays.y, arrays.m):
        passing = [b for b, yi, mi in zip(budgets, y, m) if mi and yi]
        chosen.append(passing[0] if passing else fallback_nfe)
    return float(np.mean(chosen))


def score_seed(
    model, val: SplitArrays, test: SplitArrays, budgets: tuple[int, ...], *,
    fixed_nfe: int, fallback_nfe: int, thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    cost_models: Sequence[CostModel] = (), policies: Sequence[str] = DEFAULT_POLICIES,
) -> dict:
    """Test-set metrics for one trained model. Thresholds and policies come from validation only."""
    p_val, p_test = probabilities(model, val), probabilities(model, test)
    threshold = choose_threshold(p_val, val, budgets, fixed_nfe=fixed_nfe,
                                 fallback_nfe=fallback_nfe, thresholds=thresholds)
    result = {"test_loss": _mean_loss(model, test), "test_brier": brier(p_test, test, budgets),
              "threshold": threshold, "selection": None}
    if threshold is not None:
        _, y, m = test.tensors()
        result["selection"] = compare_fixed_nfe(p_test, y, m, fixed_nfe=fixed_nfe, threshold=threshold,
                                                fallback_nfe=fallback_nfe, budgets=budgets)
    result["sequence_selection"] = [
        select_and_score(p_val, val, p_test, test, budgets, fixed_nfe=fixed_nfe, fallback_nfe=fallback_nfe,
                         cost=cost, policies=policies, thresholds=thresholds)
        for cost in cost_models
    ]
    return result
