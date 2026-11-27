"""predict_budget on the dummy bundle: budget choice, every fallback reason, metadata. No GPU, no real model."""

from __future__ import annotations

import json

import numpy as np
import pytest

from nfe_predictor import FALLBACK_REASONS, BudgetPredictor, FallbackPredictor, load_predictor
from nfe_predictor.testing import make_dummy_bundle

SHORT = "Hi there."                                                          # 2 words
MEDIUM = "The scheduler decides how many denoising steps each patch of audio receives today."  # 14 words
LONG = " ".join(["word"] * 40) + "."                                          # 40 words
HUGE = " ".join(["word"] * 200) + "."                                         # 200 words: no budget passes


@pytest.fixture
def bundle(tmp_path):
    return make_dummy_bundle(tmp_path / "b")


@pytest.fixture
def predictor(bundle):
    return BudgetPredictor.load(bundle)


@pytest.mark.parametrize("text,nfe", [(SHORT, 8), (MEDIUM, 16), (LONG, 32)])
def test_picks_the_smallest_budget_that_reaches_its_threshold(predictor, text, nfe):
    d = predictor.predict_budget({"text": text})
    assert d.nfe == nfe and d.fallback_reason is None and d.error is None
    p = d.probabilities
    budgets = [4, 8, 16, 32]
    chosen = budgets.index(nfe)
    assert p[str(nfe)] >= 0.7
    assert all(p[str(b)] < 0.7 for b in budgets[:chosen])


def test_decision_carries_versions_hash_and_time(predictor):
    d = predictor.predict_budget({"text": SHORT})
    assert d.model_version == "dummy-v0" and d.policy_version == "dummy-calib-v0"
    assert len(d.bundle_sha256) == 64 and d.decision_ms >= 0
    out = d.as_dict()
    assert set(out) == {"nfe", "fallback_reason", "model_version", "policy_version", "bundle_sha256",
                        "decision_ms", "probabilities", "error"}
    json.dumps(out)


def test_no_confident_budget_falls_back_with_probabilities(predictor):
    d = predictor.predict_budget({"text": HUGE})
    assert d.nfe == 32 and d.fallback_reason == "no_confident_budget"
    assert d.probabilities is not None and all(v < 0.7 for v in d.probabilities.values())


@pytest.mark.parametrize("context", [None, "text", {}, {"text": ""}, {"text": "   "}, {"text": 42}])
def test_invalid_input_falls_back(predictor, context):
    d = predictor.predict_budget(context)
    assert d.nfe == 32 and d.fallback_reason == "invalid_input" and d.probabilities is None


def test_text_too_long_falls_back(tmp_path):
    p = BudgetPredictor.load(make_dummy_bundle(tmp_path / "b", max_text_chars=20))
    d = p.predict_budget({"text": "x" * 21})
    assert d.nfe == 32 and d.fallback_reason == "text_too_long"


def test_feature_error_falls_back(predictor, monkeypatch):
    def boom(text):
        raise RuntimeError("tokenizer exploded")
    monkeypatch.setattr(type(predictor.bundle), "features", lambda self, text, context: boom(text))
    d = predictor.predict_budget({"text": SHORT})
    assert d.fallback_reason == "feature_error" and "tokenizer exploded" in d.error


def test_non_finite_features_fall_back(predictor, monkeypatch):
    monkeypatch.setattr(type(predictor.bundle), "features", lambda self, text, context: np.full(6, np.nan))
    assert predictor.predict_budget({"text": SHORT}).fallback_reason == "feature_error"


def test_model_error_falls_back(predictor, monkeypatch):
    def boom(x):
        raise RuntimeError("bad weights")
    monkeypatch.setattr(predictor.bundle.model, "predict_proba", boom)
    d = predictor.predict_budget({"text": SHORT})
    assert d.nfe == 32 and d.fallback_reason == "model_error"


@pytest.mark.parametrize("output", [np.array([0.9, 0.9]), np.array([np.nan, 1, 1, 1]),
                                    np.array([1.5, 1, 1, 1]), np.array([-0.1, 1, 1, 1])])
def test_invalid_prediction_falls_back(predictor, monkeypatch, output):
    monkeypatch.setattr(predictor.bundle.model, "predict_proba", lambda x: output)
    d = predictor.predict_budget({"text": SHORT})
    assert d.nfe == 32 and d.fallback_reason == "invalid_prediction" and d.probabilities is None


def test_unexpected_bug_is_internal_error_not_an_exception(predictor, monkeypatch):
    monkeypatch.setattr(predictor.bundle.policy.__class__, "choose", lambda self, p: 1 / 0)
    d = predictor.predict_budget({"text": SHORT})
    assert d.nfe == 32 and d.fallback_reason == "internal_error"


def test_every_reason_used_is_documented():
    assert len(set(FALLBACK_REASONS)) == len(FALLBACK_REASONS) == 8


def test_load_predictor_falls_back_when_the_bundle_is_broken(tmp_path):
    p = load_predictor(tmp_path / "missing", fallback_nfe=16)
    assert isinstance(p, FallbackPredictor) and p.budgets == (16,)
    d = p.predict_budget({"text": SHORT})
    assert d.nfe == 16 and d.fallback_reason == "predictor_unavailable" and "not found" in d.error


def test_load_predictor_returns_the_real_predictor(bundle):
    p = load_predictor(bundle, fallback_nfe=32)
    assert isinstance(p, BudgetPredictor) and p.budgets == (4, 8, 16, 32) and p.fallback_nfe == 32


def test_same_bundle_same_hash_and_decisions(tmp_path):
    a = BudgetPredictor.load(make_dummy_bundle(tmp_path / "a"))
    b = BudgetPredictor.load(make_dummy_bundle(tmp_path / "b"))
    assert a.bundle.sha256 == b.bundle.sha256
    for text in (SHORT, MEDIUM, LONG, HUGE):
        da, db = a.predict_budget({"text": text}), b.predict_budget({"text": text})
        assert (da.nfe, da.probabilities) == (db.nfe, db.probabilities)


def test_changing_thresholds_changes_hash_and_decision(tmp_path):
    strict = BudgetPredictor.load(make_dummy_bundle(tmp_path / "s", thresholds=(0.99, 0.99, 0.99, 0.99)))
    loose = BudgetPredictor.load(make_dummy_bundle(tmp_path / "l"))
    assert strict.bundle.sha256 != loose.bundle.sha256
    assert strict.predict_budget({"text": SHORT}).nfe > loose.predict_budget({"text": SHORT}).nfe


def test_decision_is_fast(predictor):
    predictor.predict_budget({"text": MEDIUM})
    times = [predictor.predict_budget({"text": MEDIUM}).decision_ms for _ in range(50)]
    assert sorted(times)[25] < 5.0  # median well under a millisecond in practice; generous for CI
