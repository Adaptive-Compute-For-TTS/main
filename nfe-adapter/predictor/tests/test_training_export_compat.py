"""The training side's export layout (feature/nfe-predictor-training, training/train.py save_artifacts)
loads with this package's bundle loader.

Covers the file layout: extra metadata/preprocessing keys, a 6 -> 64 -> 32 -> 9 MLP, 9 model budgets
with a policy serving a subset. Feature parity is in test_features.py.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from nfe_predictor import BudgetPredictor, BundleError, load_bundle

TRAINING_FEATURES = ("character_count", "word_count", "punctuation_count", "digit_count",
                     "average_word_length", "patch_index")
TRAINING_BUDGETS = [4, 6, 8, 10, 12, 16, 20, 24, 32]
SCHEMA = "text5-plus-patch-index-v1"


@pytest.fixture
def training_export(tmp_path):
    """A folder shaped like save_artifacts() output (minus checkpoint.pt), random weights."""
    d = tmp_path / "export"
    d.mkdir()
    rng = np.random.default_rng(0)
    dims = [6, 64, 32, 9]
    np.savez(d / "model.npz", **{k: v for i in range(3) for k, v in (
        (f"W{i}", rng.normal(size=(dims[i], dims[i + 1]))), (f"b{i}", rng.normal(size=dims[i + 1])))})
    (d / "metadata.json").write_text(json.dumps({
        "schema_version": 1, "model_type": "mlp_npz", "model_version": "patch-mlp-test",
        "prediction_unit": "audio_patch", "deployment_ready": False,
        "output_semantics": "independent_pass_probabilities", "source_schema_version": "patch-nfe-v2",
        "feature_names": list(TRAINING_FEATURES), "budgets": TRAINING_BUDGETS,
        "label_protocol": "test", "source_sha256": "0" * 64}))
    (d / "preprocessing.json").write_text(json.dumps({
        "extractor": SCHEMA, "feature_names": list(TRAINING_FEATURES),
        "mean": [10.0, 2.0, 1.0, 0.0, 4.0, 20.0], "std": [5.0, 1.0, 1.0, 1.0, 1.0, 10.0],
        "fit_split": "train", "fit_comparison_ids": ["c1", "c2"]}))
    return d


def write_policy(d, **over):
    policy = {"policy_version": "test", "budgets": TRAINING_BUDGETS, "thresholds": [0.9] * 9,
              "fallback_nfe": 32, **over}
    (d / "policy.json").write_text(json.dumps(policy))


def test_raw_export_has_no_policy_yet(training_export):
    # Training deliberately writes no policy.json until thresholds/fallback are calibrated.
    with pytest.raises(BundleError, match="missing policy.json"):
        load_bundle(training_export)


def test_export_plus_policy_loads(training_export):
    write_policy(training_export)
    b = load_bundle(training_export)
    assert b.model.n_inputs == 6 and b.model.n_outputs == 9
    assert b.policy.budgets == tuple(TRAINING_BUDGETS)
    assert b.metadata["prediction_unit"] == "audio_patch"
    p = b.model.predict_proba(np.zeros(6))
    assert p.shape == (9,) and np.all((p >= 0) & (p <= 1))


def test_policy_can_serve_a_subset_of_the_heads(training_export):
    # Pilot labels only 4/8/16/32: the policy serves those, the other 5 heads are never read.
    write_policy(training_export, budgets=[4, 8, 16, 32], thresholds=[0.9] * 4)
    b = load_bundle(training_export)
    assert b.model_budgets == tuple(TRAINING_BUDGETS) and b.heads == (0, 2, 5, 8)
    d = BudgetPredictor(b).predict_budget({"text": "Hello, world!", "patch_index": 3})
    assert set(d.probabilities) == {"4", "8", "16", "32"} and d.nfe in (4, 8, 16, 32)


def test_policy_budget_without_a_head_is_refused(training_export):
    write_policy(training_export, budgets=[4, 8, 14, 32], thresholds=[0.9] * 4)
    with pytest.raises(BundleError, match=r"budgets \[14\] have no model output"):
        load_bundle(training_export)
