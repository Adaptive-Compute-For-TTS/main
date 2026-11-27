"""Patch-level predict_budget on the dummy patch bundle (training's export layout, policy 4/8/16/32)."""

from __future__ import annotations

import pytest

from nfe_predictor import BudgetPredictor, load_predictor
from nfe_predictor.testing import make_dummy_patch_bundle

SHORT = "Hi there."                                                          # 2 words
MEDIUM = "The scheduler decides how many denoising steps each patch of audio receives today."  # 13 words
LONG = " ".join(["word"] * 40) + "."
HUGE = " ".join(["word"] * 200) + "."


@pytest.fixture
def predictor(tmp_path):
    return BudgetPredictor.load(make_dummy_patch_bundle(tmp_path / "p"))


def test_it_is_a_patch_level_predictor(predictor):
    assert predictor.prediction_unit == "audio_patch"
    assert predictor.budgets == (4, 8, 16, 32) and predictor.bundle.heads == (0, 2, 5, 8)


@pytest.mark.parametrize("text,patch_index,nfe", [
    (SHORT, 0, 16), (SHORT, 1, 8), (SHORT, 20, 8),  # first patch is harder in the dummy
    (MEDIUM, 0, 32), (MEDIUM, 3, 16),
    (LONG, 5, 32),
])
def test_nfe_depends_on_text_and_patch(predictor, text, patch_index, nfe):
    d = predictor.predict_budget({"text": text, "patch_index": patch_index})
    assert d.nfe == nfe and d.fallback_reason is None


def test_unserved_heads_are_never_read(predictor):
    # Heads 6/10/12/20/24 always say "pass" in the dummy; if read, the choice would be 6 or 10.
    for i in range(30):
        d = predictor.predict_budget({"text": SHORT, "patch_index": i})
        assert d.nfe in (4, 8, 16, 32) and set(d.probabilities) == {"4", "8", "16", "32"}


def test_no_confident_budget(predictor):
    d = predictor.predict_budget({"text": HUGE, "patch_index": 3})
    assert d.nfe == 32 and d.fallback_reason == "no_confident_budget"


@pytest.mark.parametrize("context", [{"text": SHORT}, {"text": SHORT, "patch_index": -1},
                                     {"text": SHORT, "patch_index": "0"}])
def test_missing_or_bad_patch_index_is_invalid_input(predictor, context):
    d = predictor.predict_budget(context)
    assert d.nfe == 32 and d.fallback_reason == "invalid_input" and "patch_index" in d.error


def test_load_predictor_reports_unit(tmp_path):
    assert load_predictor(make_dummy_patch_bundle(tmp_path / "p"), fallback_nfe=32).prediction_unit == "audio_patch"
    assert load_predictor(tmp_path / "missing", fallback_nfe=32).prediction_unit == "request"


def test_per_patch_decision_is_fast(predictor):
    predictor.predict_budget({"text": LONG, "patch_index": 0})
    times = sorted(predictor.predict_budget({"text": LONG, "patch_index": i}).decision_ms for i in range(200))
    assert times[100] < 1.0  # median; a request makes one call per patch
    print(f"median {times[100]:.4f} ms, p99 {times[198]:.4f} ms")
