"""GpuWorker.run_job with a fake dots.tts runtime: no weights, no GPU.

The fake drives the scheduler / executor hooks once per patch like DotsTtsModel does, so the real
generate_fixed_nfe and the real wav-validation run. The real-model check is scripts/run_one.py.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest
import torch

from gpu_worker import GpuWorker, Job, WorkerConfig

SR = 48000


def speech_like(seconds: float = 1.5) -> torch.Tensor:
    """Passes validation: 0.2 s silence, a tone, 0.2 s silence. Shape (1, samples) like dots.tts."""
    n, pad = int(seconds * SR), int(0.2 * SR)
    t = torch.arange(n - 2 * pad) / SR
    tone = 0.3 * torch.sin(2 * torch.pi * 220 * t)
    return torch.cat([torch.zeros(pad), tone, torch.zeros(pad)]).unsqueeze(0)


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def decode_patch(self, *, nfe, **kwargs):
        self.calls.append(nfe)


class FakeRuntime:
    max_generate_length = 500

    def __init__(self, num_patches: int = 6, audio=None, error: Exception | None = None,
                 ignore_scheduler: bool = False) -> None:
        self.model = types.SimpleNamespace(nfe_scheduler=None, dit_executor=None)
        self.num_patches = num_patches
        self.audio = speech_like() if audio is None else audio
        self.error = error
        self.ignore_scheduler = ignore_scheduler
        self.calls: list[dict] = []

    def resolve_sampling_options(self, *, ode_method, num_steps, guidance_scale):
        return ode_method or "euler", num_steps, guidance_scale

    def generate(self, **kwargs):
        self.calls.append({**kwargs, "seed": torch.initial_seed()})
        if self.error is not None:
            raise self.error
        for i in range(self.num_patches):
            nfe = kwargs["num_steps"]
            if self.model.nfe_scheduler is not None and not self.ignore_scheduler:
                nfe = self.model.nfe_scheduler.next_nfe(patch_index=i, default_nfe=nfe, llm_hidden=None)
            if self.model.dit_executor is not None:
                self.model.dit_executor.decode_patch(nfe=nfe)
        return {"audio": self.audio, "sample_rate": SR, "time_used": 0.75,
                "rtf": 0.75 / (self.audio.shape[-1] / SR)}


def make_worker(tmp_path: Path, **rt_kwargs):
    rt, ex = FakeRuntime(**rt_kwargs), FakeExecutor()
    cfg = WorkerConfig(storage_dir=tmp_path / "storage", warmup_nfes=(4, 10))
    return GpuWorker(rt, cfg, executor=ex), rt, ex


def test_completed_run_stores_wav_and_fills_the_row(tmp_path):
    worker, rt, ex = make_worker(tmp_path)
    out = worker.run_job(Job("run-1", "Hello there.", 10))

    assert out.ok and out.status == "completed" and out.error_message is None
    assert out.audio_path == str((tmp_path / "storage" / "run-1" / "audio.wav").resolve())
    assert Path(out.audio_path).is_file()
    assert json.loads((tmp_path / "storage" / "run-1" / "validation.json").read_text())["ok"] is True

    r = out.result
    assert r["nfe"] == 10 and r["nfe_per_patch"] == [10] * 6 and r["num_patches"] == 6
    assert ex.calls == [10] * 6 and r["executor_used_all_patches"] is True
    assert r["audio_s"] == 1.5 and r["sample_rate"] == SR and r["time_s"] == 0.75
    assert r["validation"]["ok"] is True and r["validation"]["checks"]["num_patches"] == 6
    assert r["validation"]["checks"]["max_generate_length"] == 500
    assert r["telemetry"] is None and r["quality"] is None
    json.dumps(r)  # the row's result column is JSON


def test_generate_gets_the_worker_settings_and_seed(tmp_path):
    worker, rt, _ = make_worker(tmp_path)
    torch.manual_seed(123)
    worker.run_job(Job("run-1", "Hello there.", 4))
    call = rt.calls[-1]
    assert call["text"] == "Hello there." and call["num_steps"] == 4
    assert call["seed"] == 42  # seeded per run, whatever the global state was
    assert call["guidance_scale"] == 1.2 and call["speaker_scale"] == 1.5
    assert call["prompt_audio_path"] is None and call["prompt_text"] is None


def test_hooks_are_restored_after_a_run(tmp_path):
    worker, rt, _ = make_worker(tmp_path)
    worker.run_job(Job("run-1", "Hello there.", 10))
    assert rt.model.nfe_scheduler is None and rt.model.dit_executor is None


def test_bad_audio_fails_the_run_and_goes_to_rejected(tmp_path):
    worker, _, _ = make_worker(tmp_path, audio=torch.zeros(1, SR * 2))
    out = worker.run_job(Job("run-1", "Hello there.", 10))

    assert out.status == "failed" and out.audio_path is None
    assert out.error_message == "WAV failed validation: silent"
    assert out.result["validation"]["flags"] == ["silent"]  # measurements kept for debugging
    assert (tmp_path / "storage" / "_rejected" / "run-1" / "audio.wav").is_file()
    assert not (tmp_path / "storage" / "run-1").exists()


def test_nfe_mismatch_fails_the_run(tmp_path):
    # The runtime ignores the scheduler: patches get num_steps, but generate_fixed_nfe still checks
    # its own log, which then stays empty -> "No patches were decoded".
    worker, _, _ = make_worker(tmp_path, ignore_scheduler=True)
    out = worker.run_job(Job("run-1", "Hello there.", 10))
    assert out.status == "failed" and out.result is None
    assert out.error_message.startswith("RuntimeError: No patches were decoded")


def test_generation_error_fails_the_run_without_raising(tmp_path):
    worker, _, _ = make_worker(tmp_path, error=ValueError("bad text"))
    out = worker.run_job(Job("run-1", "Hello there.", 10))
    assert out.status == "failed" and out.error_message == "ValueError: bad text"
    assert not (tmp_path / "storage" / "run-1").exists()


def test_out_of_memory_has_a_clear_message(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    worker, _, _ = make_worker(tmp_path, error=torch.cuda.OutOfMemoryError("CUDA out of memory. Tried ..."))
    out = worker.run_job(Job("run-1", "Hello there.", 10))
    assert out.status == "failed" and out.error_message == "out of GPU memory"


def test_same_run_twice_fails_instead_of_overwriting(tmp_path):
    worker, _, _ = make_worker(tmp_path)
    assert worker.run_job(Job("run-1", "Hello there.", 10)).ok
    out = worker.run_job(Job("run-1", "Hello there.", 10))
    assert out.status == "failed" and "already stored" in out.error_message


@pytest.mark.parametrize("nfe", [0, -3, True, 10.0, "10"])
def test_invalid_nfe_fails_before_generating(tmp_path, nfe):
    worker, rt, _ = make_worker(tmp_path)
    out = worker.run_job(Job("run-1", "Hello there.", nfe))
    assert out.status == "failed" and out.error_message.startswith("invalid nfe")
    assert rt.calls == []


def test_empty_prompt_fails_before_generating(tmp_path):
    worker, rt, _ = make_worker(tmp_path)
    assert worker.run_job(Job("run-1", "   ", 10)).error_message == "empty prompt"
    assert rt.calls == []


def test_unsafe_run_id_fails(tmp_path):
    worker, _, _ = make_worker(tmp_path)
    out = worker.run_job(Job("../escape", "Hello there.", 10))
    assert out.status == "failed" and "unsafe run_id" in out.error_message
    assert not (tmp_path / "escape").exists()


def test_warmup_runs_each_nfe_and_stores_nothing(tmp_path):
    worker, rt, _ = make_worker(tmp_path)
    worker.warmup()
    assert [c["num_steps"] for c in rt.calls] == [4, 10]
    assert not (tmp_path / "storage").exists()
