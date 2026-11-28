"""Switch-aware cost of a budget choice, for comparing representations on speed, not just mean NFE.

On dots.tts's optimized path the DiT KV cache holds one K/V copy per ODE step for one NFE, so
changing NFE between patches rebuilds it (about 8x a normal patch; see
gpu-executor/docs/design/mixed-nfe-kv-cache.md on nfe-predictor-inference). A predictor that saves
NFE but switches often can be slower than fixed NFE 32. Here each request's patches are chosen in
order by a policy, and the request is priced with a cost model.

Caveat: reviewed patch labels come from one fixed anchor state per patch, not from the trajectory an
adaptive schedule would actually produce. Pass rates here are per-patch estimates, as in
training.evaluate.compare_fixed_nfe; costs are estimates from one GPU and one utterance.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import re

import numpy as np
import torch

from models.mlp import choose_budget


@dataclass(frozen=True)
class CostModel:
    name: str
    patch_ms: Callable[[int], float]   # a patch at this NFE when the previous patch used the same NFE
    switch_ms: Callable[[int], float]  # a patch at this NFE right after a change of NFE
    source: str
    measured_nfes: tuple[int, ...] | None = None  # None: exact for every NFE; else others are estimated

    def request_ms(self, nfes: Sequence[int]) -> float:
        return sum(self.patch_ms(n) if i == 0 or n == nfes[i - 1] else self.switch_ms(n)
                   for i, n in enumerate(nfes))


def _measured(points: dict[int, float]) -> Callable[[int], float]:
    """Piecewise-linear through measured NFEs; proportional to NFE below the smallest one."""
    xs = sorted(points)
    ys = [points[x] for x in xs]

    def cost(nfe: int) -> float:
        if nfe < xs[0]:
            return ys[0] * nfe / xs[0]
        return float(np.interp(nfe, xs, ys))
    return cost


COST_MODELS = {
    "nfe": CostModel("nfe", float, float, "cost = NFE, no switch cost (the plain mean-NFE proxy)"),
    "eager-rtx5060": CostModel(
        "eager-rtx5060", lambda n: 16.0 * n, lambda n: 16.0 * n,
        "eager path, ~16 ms per DiT step, switching free; RTX 5060 laptop, dots.tts-soar bf16, 2026-11-26",
        (8, 16, 32)),
    "optimized-rtx5060": CostModel(
        "optimized-rtx5060", _measured({8: 49.0, 16: 124.0, 32: 307.0}),
        _measured({8: 445.0, 16: 970.0, 32: 1900.0}),
        "optimized path (KV cache + CUDA graphs) DiT ms per patch, same NFE vs after a switch; "
        "RTX 5060 laptop, dots.tts-soar bf16, one 7.4 s sentence, 2026-11-26; measured at NFE 8/16/32, "
        "interpolated between and proportional to NFE below 8", (8, 16, 32)),
}


# --- policies: per-patch choices -> the NFEs actually run ------------------------------------

def _per_patch(choices: Sequence[int]) -> list[int]:
    return list(choices)


def _request_max(choices: Sequence[int]) -> list[int]:
    return [max(choices)] * len(choices)


def _min_run(run: int) -> Callable[[Sequence[int]], list[int]]:
    """Go up at once (quality first); go down only after `run` patches at the current NFE."""
    def apply(choices: Sequence[int]) -> list[int]:
        out: list[int] = []
        held = 0  # patches run at out[-1] so far, in a row
        for c in choices:
            if not out:
                nxt = c
            else:
                current = out[-1]
                nxt = c if c > current or (c < current and held >= run) else current
            held = held + 1 if out and nxt == out[-1] else 1
            out.append(nxt)
        return out
    return apply


_MIN_RUN = re.compile(r"^min_run(\d+)$")


def policy(name: str) -> Callable[[Sequence[int]], list[int]]:
    if name == "per_patch":
        return _per_patch
    if name == "request_max":
        return _request_max
    match = _MIN_RUN.match(name)
    if match and int(match[1]) >= 1:
        return _min_run(int(match[1]))
    raise ValueError(f"unknown policy {name!r}: per_patch, request_max or min_runR (R >= 1)")


DEFAULT_POLICIES = ("per_patch", "min_run4", "request_max")


# --- evaluation --------------------------------------------------------------------------------

def _sequences(sequence_ids: Sequence[str], positions: Sequence[int]) -> list[list[int]]:
    """Row indices grouped by request, each in patch order."""
    groups: dict[str, list[int]] = defaultdict(list)
    for row, seq in enumerate(sequence_ids):
        groups[seq].append(row)
    return [sorted(rows, key=lambda r: positions[r]) for _, rows in sorted(groups.items())]


def _rate(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if len(values) else None


def run_schedule(
    run_nfes: Sequence[int], rows: Sequence[int], y: np.ndarray, m: np.ndarray,
    budgets: tuple[int, ...], sequences: Sequence[Sequence[int]], cost: CostModel, *, fixed_nfe: int,
) -> dict:
    """Cost and switches per request, and pass rates on paired patches: those with both the run
    NFE's and the fixed NFE's outcome observed (as in compare_fixed_nfe), so a policy can't look
    good by landing on budgets where hard patches went unmeasured."""
    column = {b: i for i, b in enumerate(budgets)}
    nfe_of_row = dict(zip(rows, run_nfes))
    fixed = column[fixed_nfe]
    paired = [(r, column[nfe_of_row[r]]) for r in rows if m[r, column[nfe_of_row[r]]] and m[r, fixed]]
    per_request = [[nfe_of_row[r] for r in seq] for seq in sequences]
    estimated = cost.measured_nfes is not None and any(n not in cost.measured_nfes for n in run_nfes)
    return {
        "pass_rate": _rate([y[r, c] for r, c in paired]),
        "fixed_pass_rate": _rate([y[r, fixed] for r, _ in paired]),
        "paired_coverage": len(paired) / len(rows) if len(rows) else None,
        "cost_estimated_outside_measured_nfes": estimated,
        "mean_request_ms": float(np.mean([cost.request_ms(s) for s in per_request])),
        "mean_switches": float(np.mean([sum(a != b for a, b in zip(s, s[1:])) for s in per_request])),
        "mean_nfe": float(np.mean(run_nfes)),
    }


def evaluate_policy(
    p: torch.Tensor, y: np.ndarray, m: np.ndarray, sequence_ids: Sequence[str], positions: Sequence[int],
    budgets: tuple[int, ...], *, policy_name: str, threshold: float, fallback_nfe: int, cost: CostModel,
    fixed_nfe: int,
) -> dict:
    sequences = _sequences(sequence_ids, positions)
    apply = policy(policy_name)
    rows, run_nfes = [], []
    for seq in sequences:
        choices = [choose_budget(p[r], budgets, threshold, fallback_nfe) for r in seq]
        rows.extend(seq)
        run_nfes.extend(apply(choices))
    return run_schedule(run_nfes, rows, y, m, budgets, sequences, cost, fixed_nfe=fixed_nfe)


def evaluate_fixed(
    y: np.ndarray, m: np.ndarray, sequence_ids: Sequence[str], positions: Sequence[int],
    budgets: tuple[int, ...], *, fixed_nfe: int, cost: CostModel,
) -> dict:
    sequences = _sequences(sequence_ids, positions)
    rows = [r for seq in sequences for r in seq]
    return run_schedule([fixed_nfe] * len(rows), rows, y, m, budgets, sequences, cost, fixed_nfe=fixed_nfe)


def select_and_score(
    p_val: torch.Tensor, val, p_test: torch.Tensor, test, budgets: tuple[int, ...], *,
    fixed_nfe: int, fallback_nfe: int, cost: CostModel, policies: Sequence[str], thresholds: Sequence[float],
) -> dict:
    """Pick (policy, threshold) on validation: lowest estimated ms whose pass rate is at least
    fixed NFE's on the same (paired) validation patches. Then score that one choice on test.

    `val`/`test` need .y, .m, .sequence_ids, .positions (experiments.ablation_train.SplitArrays).
    """
    best = None
    for name in policies:
        for t in sorted(thresholds):
            r = evaluate_policy(p_val, val.y, val.m, val.sequence_ids, val.positions, budgets,
                                policy_name=name, threshold=t, fallback_nfe=fallback_nfe, cost=cost,
                                fixed_nfe=fixed_nfe)
            if r["pass_rate"] is None or r["pass_rate"] < r["fixed_pass_rate"]:
                continue
            if best is None or r["mean_request_ms"] < best[2]:
                best = (name, t, r["mean_request_ms"])
    fixed_test = evaluate_fixed(test.y, test.m, test.sequence_ids, test.positions, budgets,
                                fixed_nfe=fixed_nfe, cost=cost)
    if best is None:
        return {"cost_model": cost.name, "policy": None, "threshold": None, "test": None, "fixed_test": fixed_test}
    chosen = evaluate_policy(p_test, test.y, test.m, test.sequence_ids, test.positions, budgets,
                             policy_name=best[0], threshold=best[1], fallback_nfe=fallback_nfe, cost=cost,
                             fixed_nfe=fixed_nfe)
    return {"cost_model": cost.name, "policy": best[0], "threshold": best[1], "test": chosen,
            "fixed_test": fixed_test,
            "speedup_vs_fixed": fixed_test["mean_request_ms"] / chosen["mean_request_ms"]}
