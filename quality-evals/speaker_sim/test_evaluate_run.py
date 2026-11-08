"""Tests for evaluate_run.py. No model, no torch: the scorer and ASR are replaced by fakes.

    pytest quality-evals/speaker_sim/test_evaluate_run.py
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_run import QualityEvaluator, normalize, wer_cer  # noqa: E402


class FakeSim:
    def __init__(self, score=0.93, error=None):
        self.score, self.error, self.calls = score, error, []

    def similarity(self, a, b):
        self.calls.append((a, b))
        if self.error:
            raise self.error
        return self.score


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "clip.wav"
    path.write_bytes(b"RIFF")  # the fakes never read it; the evaluator only checks it exists
    return str(path)


def test_normalize_keeps_decimals_and_apostrophes():
    assert normalize("Don't pay $42.75, ok?") == ["don't", "pay", "$42.75", "ok"]
    assert normalize("It costs 3.5.") == ["it", "costs", "3.5"]


def test_wer_cer_perfect_and_errors():
    assert wer_cer("The kettle is on.", "the kettle is on") == {"wer": 0.0, "cer": 0.0}
    r = wer_cer("the kettle is on", "the kettle was on")
    assert r["wer"] == 0.25 and 0 < r["cer"] < 0.25


def test_wer_is_capped_at_one():
    assert wer_cer("hi", "a b c d e f")["wer"] == 1.0


def test_wer_empty_reference_raises():
    with pytest.raises(ValueError):
        wer_cer("...", "hello")


def test_evaluate_scores_a_clip(wav):
    sim = FakeSim(0.9321)
    qe = QualityEvaluator("ref.wav", sim=sim, transcribe=lambda p: "The kettle is on.")
    q = qe.evaluate(wav, "The kettle is on.", validation={"ok": True, "flags": []})
    assert q["scored"] and q["sim_ref"] == 0.9321 and q["wer"] == 0.0 and q["errors"] == []
    assert sim.calls == [(wav, "ref.wav")]
    json.dumps(q)  # must be storable in the backend's JSON column


def test_failed_validation_is_not_scored(wav):
    sim = FakeSim()
    qe = QualityEvaluator("ref.wav", sim=sim, transcribe=lambda p: "x")
    q = qe.evaluate(wav, "x", validation={"ok": False, "flags": ["silent"]})
    assert not q["scored"] and sim.calls == [] and "silent" in q["errors"][0]


def test_missing_audio(tmp_path):
    qe = QualityEvaluator("ref.wav", sim=FakeSim(), transcribe=lambda p: "x")
    q = qe.evaluate(str(tmp_path / "nope.wav"), "x")
    assert not q["scored"] and q["errors"] == ["audio not found: nope.wav"]


def test_one_failing_scorer_keeps_the_other_metric(wav):
    qe = QualityEvaluator("ref.wav", sim=FakeSim(error=ValueError("too short")), transcribe=lambda p: "hello there")
    q = qe.evaluate(wav, "hello there")
    assert q["sim_ref"] is None and q["wer"] == 0.0 and q["scored"]
    assert q["errors"] == ["sim_ref: ValueError: too short"]


def test_asr_failure_is_recorded_not_raised(wav):
    def boom(path):
        raise RuntimeError("out of memory")

    q = QualityEvaluator("ref.wav", sim=FakeSim(), transcribe=boom).evaluate(wav, "hello")
    assert q["sim_ref"] == 0.93 and q["wer"] is None and q["errors"] == ["asr: RuntimeError: out of memory"]


def test_no_reference_skips_sim(wav):
    sim = FakeSim()
    q = QualityEvaluator(None, sim=sim, transcribe=lambda p: "hi").evaluate(wav, "hi")
    assert q["sim_ref"] is None and sim.calls == [] and q["wer"] == 0.0
