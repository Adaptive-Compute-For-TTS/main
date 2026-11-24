"""Masked offline evaluation; no quality thresholds or serving policy are approved here."""
from __future__ import annotations

import math

import torch

from models.mlp import choose_budget, masked_bce_loss
from training.contracts import CANDIDATE_NFES, validate_budgets


def _validate_predictions(probabilities, targets, mask, budgets):
    validate_budgets(budgets)
    if (probabilities.ndim != 2 or probabilities.shape[1] != len(budgets)
            or probabilities.shape != targets.shape or targets.shape != mask.shape):
        raise ValueError("Expected matching [patches, budgets] probability, target and mask tensors")
    if not torch.all(torch.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1)):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Masks must be binary")
    if not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("Prediction metrics require binary pass/fail targets")


def per_budget_metrics(probabilities, targets, mask, budgets=CANDIDATE_NFES, threshold=0.5):
    """Diagnostic classification metrics, restricted to observed patch outcomes.

    The classification cutoff is not a quality criterion or calibrated NFE policy.
    Undefined metrics are None rather than invented zero scores.
    """
    _validate_predictions(probabilities, targets, mask, budgets)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Classification threshold must be in [0, 1]")
    result = {}
    for i, budget in enumerate(budgets):
        known = mask[:, i].bool()
        p, y = probabilities[known, i], targets[known, i].bool()
        predicted = p >= threshold
        count = int(known.sum())
        tp = int((predicted & y).sum())
        tn = int((~predicted & ~y).sum())
        fp = int((predicted & ~y).sum())
        fn = int((~predicted & y).sum())
        result[str(budget)] = {
            "observed": count, "passes": int(y.sum()), "failures": count - int(y.sum()),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": (tp + tn) / count if count else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "brier": float(((p - y.float()) ** 2).mean()) if count else None,
        }
    return result


def fixed_nfe_baseline(targets, mask, fixed_nfe: int, budgets=CANDIDATE_NFES):
    """Observed pass fraction for a fixed budget, with explicit missing-label coverage.

    This describes controlled patch outcomes, not whole-utterance audio quality.
    NFE is a compute proxy; this function makes no latency claim.
    """
    _validate_predictions(torch.zeros_like(targets), targets, mask, budgets)
    if type(fixed_nfe) is not int or fixed_nfe not in budgets:
        raise ValueError("Fixed baseline must be a candidate budget")
    column = budgets.index(fixed_nfe)
    known = mask[:, column].bool()
    count = int(known.sum())
    total = len(targets)
    return {
        "nfe": fixed_nfe, "patches": total, "observed": count,
        "coverage": count / total if total else None,
        "pass_rate": float(targets[known, column].float().mean()) if count else None,
    }


def compare_fixed_nfe(
    probabilities, targets, mask, *, fixed_nfe: int, threshold: float,
    fallback_nfe: int, budgets=CANDIDATE_NFES,
):
    """Compare an explicitly supplied offline policy with fixed NFE on paired labels.

    Both selected outcomes must be measured for a patch to enter the comparison.
    Report coverage because a partial-label subset may be biased. No interpolation
    of unmeasured outcomes and no end-to-end adaptive-generation claims are made.
    """
    _validate_predictions(probabilities, targets, mask, budgets)
    baseline = fixed_nfe_baseline(targets, mask, fixed_nfe, budgets)
    if (not math.isfinite(threshold) or not 0 <= threshold <= 1
            or type(fallback_nfe) is not int or fallback_nfe not in budgets):
        raise ValueError("Supply an explicit valid threshold and candidate fallback")
    choices = [choose_budget(p, budgets, threshold, fallback_nfe) for p in probabilities]
    cols = torch.tensor([budgets.index(n) for n in choices], dtype=torch.long, device=targets.device)
    rows = torch.arange(len(targets), device=targets.device)
    fixed_col = budgets.index(fixed_nfe)
    paired = mask[rows, cols].bool() & mask[:, fixed_col].bool()
    count = int(paired.sum())
    selected_nfes = targets.new_tensor(choices)
    return {
        "fixed": baseline, "threshold": threshold, "fallback_nfe": fallback_nfe,
        "paired_observed": count,
        "paired_coverage": count / len(targets) if len(targets) else None,
        "adaptive_pass_rate": float(targets[rows, cols][paired].float().mean()) if count else None,
        "fixed_pass_rate": float(targets[paired, fixed_col].float().mean()) if count else None,
        "adaptive_mean_nfe": float(selected_nfes[paired].float().mean()) if count else None,
        "fixed_mean_nfe": float(fixed_nfe) if count else None,
    }


@torch.no_grad()
def evaluate(model, loader, budgets=CANDIDATE_NFES, threshold=0.5, fixed_nfe=None):
    """Return observation-weighted BCE and metrics, preserving the model's mode."""
    was_training = model.training
    model.eval()
    probabilities, targets, masks = [], [], []
    loss_sum, observed = 0.0, 0
    device = next(model.parameters()).device
    try:
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            logits = model(x)
            if logits.ndim != 2 or logits.shape[1] != len(budgets):
                raise ValueError("Model output does not match the dataset budget axis")
            loss = masked_bce_loss(logits, y, mask)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite evaluation loss")
            count = int(mask.sum())
            loss_sum += float(loss) * count
            observed += count
            probabilities.append(logits.sigmoid().cpu())
            targets.append(y.cpu())
            masks.append(mask.cpu())
    finally:
        model.train(was_training)
    if not observed:
        raise ValueError("Evaluation requires observed labels")
    p, y, mask = torch.cat(probabilities), torch.cat(targets), torch.cat(masks)
    result = {
        "loss": loss_sum / observed, "observed": observed, "patches": len(y),
        "classification_threshold": threshold,
        "per_budget": per_budget_metrics(p, y, mask, budgets, threshold),
    }
    if fixed_nfe is not None:
        result["fixed_nfe_baseline"] = fixed_nfe_baseline(y, mask, fixed_nfe, budgets)
    return result
