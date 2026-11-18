"""load_bundle refuses bundles whose parts don't fit together, at startup rather than per request."""

from __future__ import annotations

import json

import numpy as np
import pytest

from nfe_predictor import BundleError, load_bundle
from nfe_predictor.testing import make_dummy_bundle


@pytest.fixture
def bundle(tmp_path):
    return make_dummy_bundle(tmp_path / "b")


def edit(path, name, **changes):
    data = json.loads((path / name).read_text())
    data.update(changes)
    (path / name).write_text(json.dumps(data))


def test_valid_bundle_loads(bundle):
    b = load_bundle(bundle)
    assert b.model_version == "dummy-v0" and b.policy.budgets == (4, 8, 16, 32)
    assert b.features("Hi there.").shape == (6,)


@pytest.mark.parametrize("name", ["metadata.json", "preprocessing.json", "policy.json", "model.npz"])
def test_missing_file(bundle, name):
    (bundle / name).unlink()
    with pytest.raises(BundleError, match="missing"):
        load_bundle(bundle)


def test_not_a_folder(tmp_path):
    with pytest.raises(BundleError, match="not found"):
        load_bundle(tmp_path / "nope")


def test_bad_json(bundle):
    (bundle / "policy.json").write_text("{not json")
    with pytest.raises(BundleError, match="not valid JSON"):
        load_bundle(bundle)


@pytest.mark.parametrize("changes,match", [
    ({"schema_version": 99}, "schema_version"),
    ({"model_type": "sklearn_joblib"}, "unknown model_type"),
])
def test_bad_metadata(bundle, changes, match):
    edit(bundle, "metadata.json", **changes)
    with pytest.raises(BundleError, match=match):
        load_bundle(bundle)


def test_feature_names_must_match_the_extractor_in_order(bundle):
    names = json.loads((bundle / "preprocessing.json").read_text())["feature_names"]
    edit(bundle, "preprocessing.json", feature_names=names[::-1])
    with pytest.raises(BundleError, match="training/inference mismatch"):
        load_bundle(bundle)


def test_unknown_extractor(bundle):
    edit(bundle, "preprocessing.json", extractor="shivani_v9")
    with pytest.raises(BundleError, match="unknown feature extractor"):
        load_bundle(bundle)


@pytest.mark.parametrize("changes,match", [
    ({"std": [1, 1, 1, 1, 1, 0]}, "std"),
    ({"mean": [0, 0, 0]}, "mean/std need 6"),
])
def test_bad_scaling(bundle, changes, match):
    edit(bundle, "preprocessing.json", **changes)
    with pytest.raises(BundleError, match=match):
        load_bundle(bundle)


@pytest.mark.parametrize("changes,match", [
    ({"budgets": [8, 4, 16, 32]}, "ascending"),
    ({"budgets": [4, 4, 16, 32]}, "ascending"),
    ({"thresholds": [0.9, 0.9, 0.9]}, "threshold"),
    ({"thresholds": [0.9, 0.9, 0.9, 0]}, "threshold"),
    ({"fallback_nfe": 10}, "not one of the budgets"),
    ({"budgets": [4, 8, 16]}, "threshold"),
])
def test_bad_policy(bundle, changes, match):
    edit(bundle, "policy.json", **changes)
    with pytest.raises(BundleError, match=match):
        load_bundle(bundle)


def test_model_outputs_must_match_budgets(bundle):
    edit(bundle, "policy.json", budgets=[4, 8, 16], thresholds=[0.7, 0.7, 0.7], fallback_nfe=16)
    with pytest.raises(BundleError, match="3 budgets"):
        load_bundle(bundle)


def test_model_inputs_must_match_features(bundle):
    np.savez(bundle / "model.npz", W0=np.zeros((5, 4)), b0=np.zeros(4))
    with pytest.raises(BundleError, match="takes 5 inputs"):
        load_bundle(bundle)


def test_inconsistent_layers(bundle):
    np.savez(bundle / "model.npz", W0=np.zeros((6, 3)), b0=np.zeros(3), W1=np.zeros((2, 4)), b1=np.zeros(4))
    with pytest.raises(BundleError, match="could not load"):
        load_bundle(bundle)


def test_pickled_weights_are_refused(bundle):
    np.savez(bundle / "model.npz", W0=np.array([object()], dtype=object), b0=np.zeros(4))
    with pytest.raises(BundleError, match="could not load"):
        load_bundle(bundle)
