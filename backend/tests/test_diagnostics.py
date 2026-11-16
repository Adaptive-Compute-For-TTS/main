from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
from backend.app.workers import diagnostics


def test_real_cpu_telemetry_is_json_safe_and_generates_once():
    import json
    calls = []
    value, report = diagnostics.GenerationTelemetry().measure(lambda: calls.append(1) or "audio", 4)
    assert calls == [1] and value == "audio"
    assert report["scope"] == "generation_only" and report["status"] == "collected"
    assert report["timing"]["repeats"] == 1 and report["timing"]["wall_time_s"] >= 0
    if report["fingerprint"]["device"] == "cpu":
        assert report["memory"]["peak_allocated_mb"] == "n/a"
        assert report["gpu"]["num_samples"] == 0
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("after_generation", [False, True])
def test_instrumentation_failure_never_repeats_generation(monkeypatch, after_generation):
    original = diagnostics._telemetry_module
    def time_callable(fn, *args, **kwargs):
        if after_generation:
            fn()
        raise RuntimeError("timer unavailable")
    monkeypatch.setattr(diagnostics, "_telemetry_module", lambda name: SimpleNamespace(time_callable=time_callable) if name == "timing" else original(name))
    calls = []
    value, report = diagnostics.GenerationTelemetry().measure(lambda: calls.append(1) or "audio", 4)
    assert calls == [1] and value == "audio"
    assert report["status"] == "partial" and "timer unavailable" in report["errors"][0]


def test_generation_error_is_not_retried_or_hidden():
    calls = []
    def generate():
        calls.append(1)
        raise ValueError("generation failed")
    with pytest.raises(ValueError, match="generation failed"):
        diagnostics.GenerationTelemetry().measure(generate, 4)
    assert calls == [1]


def test_quality_uses_existing_evaluator_and_records_partial_results(monkeypatch):
    scorer = diagnostics.RunQuality()
    scorer.evaluator = SimpleNamespace(evaluate=lambda *args, **kwargs: {"scored": True, "wer": 0.25, "cer": 0.1, "sim_ref": None, "errors": ["reference unavailable"]})
    result = scorer("audio.wav", "hello world", {"ok": True})
    assert result["wer"] == 0.25 and result["status"] == "partial"
    assert "no reference" in result["notes"][0]


def test_quality_import_failure_is_visible_and_invalid_wavs_skip_loading(monkeypatch):
    scorer = diagnostics.RunQuality()
    def fail():
        raise ImportError("faster_whisper unavailable")
    monkeypatch.setattr(scorer, "_load", fail)
    result = scorer("audio.wav", "hello", {"ok": True})
    assert result["status"] == "unavailable" and not result["scored"]
    assert "faster_whisper" in result["errors"][0]
    assert scorer("audio.wav", "hello", {"ok": False})["status"] == "skipped"
