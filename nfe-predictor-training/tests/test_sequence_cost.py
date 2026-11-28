"""Switch-aware cost and policies, on hand-built schedules; not experimental results."""
import numpy as np
import pytest
import torch

from experiments.ablation_data import from_patch_targets
from experiments.ablation_train import SplitArrays
from experiments.feature_ablation import run_ablation, to_markdown
from experiments.sequence_cost import COST_MODELS, evaluate_policy, policy, select_and_score
from training.train import TrainingConfig

OPT = COST_MODELS["optimized-rtx5060"]


def test_optimized_cost_model_prices_switches():
    assert OPT.request_ms([8, 8, 32]) == 49 + 49 + 1900
    assert OPT.request_ms([32] * 3) == 3 * 307
    assert OPT.patch_ms(4) == pytest.approx(24.5)    # proportional below the smallest measured NFE
    assert OPT.patch_ms(12) == pytest.approx(86.5)   # interpolated
    assert COST_MODELS["nfe"].request_ms([8, 32]) == 40
    assert COST_MODELS["eager-rtx5060"].request_ms([8, 32]) == 16 * 40


def test_policies():
    choices = [32, 8, 8, 32, 8, 8, 8]
    assert policy("per_patch")(choices) == choices
    assert policy("request_max")(choices) == [32] * 7
    # Up at once, down only after 2 patches at the current NFE.
    assert policy("min_run2")(choices) == [32, 32, 8, 32, 32, 8, 8]
    assert policy("min_run1")([8, 32, 8]) == [8, 32, 8]
    for bad in ("min_run0", "fastest", "min_runx"):
        with pytest.raises(ValueError):
            policy(bad)


def _alternating():
    """One 6-patch request: even patches pass at NFE 8, odd ones need 32. Rows out of order."""
    order = [3, 0, 5, 1, 4, 2]
    y = np.array([[float(pos % 2 == 0), 1.0] for pos in order])
    p = torch.tensor([[0.99 if pos % 2 == 0 else 0.1, 0.99] for pos in order])
    arrays = SplitArrays(np.zeros((6, 1)), y, np.ones_like(y), ("req",) * 6, tuple(order))
    return p, arrays


def test_evaluate_policy_orders_patches_and_counts_switches():
    p, a = _alternating()
    r = evaluate_policy(p, a.y, a.m, a.sequence_ids, a.positions, (8, 32), policy_name="per_patch",
                        threshold=0.5, fallback_nfe=32, cost=OPT, fixed_nfe=32)
    assert r["mean_switches"] == 5 and r["pass_rate"] == 1.0 and r["paired_coverage"] == 1.0
    assert r["mean_request_ms"] == 49 + 1900 + 445 + 1900 + 445 + 1900
    assert r["cost_estimated_outside_measured_nfes"] is False


def test_pass_rates_are_paired_with_the_fixed_budget():
    # Patch 1 fails at NFE 8 and its NFE-32 outcome is unknown: it must not count for either side.
    y = np.array([[1.0, 1.0], [0.0, 0.0]])
    m = np.array([[1.0, 1.0], [1.0, 0.0]])
    p = torch.tensor([[0.99, 0.99], [0.99, 0.99]])
    r = evaluate_policy(p, y, m, ("a", "b"), (0, 0), (8, 32), policy_name="per_patch",
                        threshold=0.5, fallback_nfe=32, cost=OPT, fixed_nfe=32)
    assert r["pass_rate"] == 1.0 and r["fixed_pass_rate"] == 1.0 and r["paired_coverage"] == 0.5
    r4 = evaluate_policy(torch.tensor([[0.99, 0.99]]), np.ones((1, 2)), np.ones((1, 2)), ("a",), (0,), (4, 32),
                         policy_name="per_patch", threshold=0.5, fallback_nfe=32, cost=OPT, fixed_nfe=32)
    assert r4["cost_estimated_outside_measured_nfes"] is True


def test_selection_depends_on_the_cost_model():
    p, a = _alternating()
    kwargs = dict(fixed_nfe=32, fallback_nfe=32, policies=("per_patch", "min_run4", "request_max"),
                  thresholds=(0.5,))
    by_nfe = select_and_score(p, a, p, a, (8, 32), cost=COST_MODELS["nfe"], **kwargs)
    by_time = select_and_score(p, a, p, a, (8, 32), cost=OPT, **kwargs)
    assert by_nfe["policy"] == "per_patch"       # mean NFE 20 looks best...
    assert by_time["policy"] == "request_max"    # ...but switching every patch is the slowest on GPU
    assert by_time["speedup_vs_fixed"] == pytest.approx(1.0)
    assert by_nfe["speedup_vs_fixed"] == pytest.approx(32 / 20)


def test_no_choice_when_nothing_matches_fixed_quality():
    p, a = _alternating()
    worse = torch.tensor([[0.99, 0.99]] * 6)  # always NFE 8: odd patches fail
    r = select_and_score(worse, a, worse, a, (8, 32), cost=OPT, fixed_nfe=32, fallback_nfe=32,
                         policies=("per_patch",), thresholds=(0.5,))
    assert r["policy"] is None and r["test"] is None and r["fixed_test"]["pass_rate"] == 1.0


def test_ablation_reports_switch_aware_selection(synthetic_examples):
    data = from_patch_targets(synthetic_examples)
    assert len({e.sequence_id for e in data.examples}) < len(data.examples)  # patches share requests
    report = run_ablation(data, ["text5"], seeds=(0,), config=TrainingConfig(epochs=1),
                          fixed_nfe=8, fallback_nfe=8, cost_models=(OPT,), policies=("per_patch", "min_run2"))
    summary = report["feature_sets"][0]["summary"]["sequence_selection"]
    assert [s["cost_model"] for s in summary] == ["optimized-rtx5060"]
    assert "switch cost" in to_markdown(report)
    with pytest.raises(ValueError, match="unknown policy"):
        run_ablation(data, ["text5"], seeds=(0,), fixed_nfe=8, fallback_nfe=8, policies=("nope",))
