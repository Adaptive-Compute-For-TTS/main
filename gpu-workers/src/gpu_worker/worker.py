"""Load dots.tts once, then turn each job into a validated, stored WAV and the run's result.

One job, following the system diagram:
    fixed-NFE executor (gpu-executor: generate_fixed_nfe, every patch at the run's NFE)
    -> runtime measurements (time, RTF, peak VRAM; telemetry plugs in here)
    -> WAV validation + storage (wav-validation: validate_and_store, check first, then store)
    -> quality eval (quality-evals: speaker similarity + WER/CER on the stored clip, passed clips only)
    -> JobOutcome, which the queue loop writes to the runs row

run_job never raises for a failed run: any error becomes status "failed" with an error_message,
so one bad run can't stop the worker.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from adaptive_nfe import generate_fixed_nfe
from wav_validation import validate_and_store

from gpu_worker import __version__
from gpu_worker.config import WorkerConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Job:
    """One run from the queue: the backend's runs.id, runs.prompt, runs.nfe."""

    run_id: str
    prompt: str
    nfe: int


@dataclass
class JobOutcome:
    """What the queue loop writes back to the runs row."""

    status: str  # "completed" or "failed" (the backend's RunStatus values)
    result: Optional[dict]  # runs.result (JSON); None if generation never finished
    audio_path: Optional[str]  # runs.audio_path; only for clips that passed validation
    error_message: Optional[str]  # runs.error_message; only when failed

    @property
    def ok(self) -> bool:
        return self.status == "completed"


def audio_sha256(audio: torch.Tensor) -> str:
    # Hash of the samples, not the WAV file (float WAVs carry a timestamped header chunk).
    return hashlib.sha256(audio.float().cpu().contiguous().numpy().tobytes()).hexdigest()


def seed_all(seed: int) -> None:
    # Same as gpu-executor/scripts/generate.py and ml-evals/run_baseline.py.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_evaluator(config: WorkerConfig):
    """The per-run quality evaluator from quality-evals/speaker_sim (models load on its first use).

    Needs quality-evals/speaker_sim/requirements-worker.txt installed on top of gpu-executor's
    constraints. It lives in the monorepo rather than on PyPI, so its folder goes on sys.path.
    """
    folder = Path(__file__).resolve().parents[3] / "quality-evals" / "speaker_sim"
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    from evaluate_run import QualityEvaluator

    reference = str(config.reference_wav) if config.reference_wav else None
    return QualityEvaluator(reference_wav=reference)


def _failed(message: str, result: Optional[dict] = None) -> JobOutcome:
    return JobOutcome(status="failed", result=result, audio_path=None, error_message=message)


class GpuWorker:
    def __init__(self, runtime, config: WorkerConfig, executor=None, evaluator=None) -> None:
        """Wrap an already loaded runtime. Use GpuWorker.load() to load dots.tts from config.

        executor: the adaptive_nfe Executor, built once and reused for every job (None builds one
        per call inside generate_fixed_nfe).
        evaluator: object with .evaluate(audio_path, prompt, validation=...) -> dict that never raises
        (quality-evals' QualityEvaluator); None skips quality eval and leaves result["quality"] empty.
        """
        self.rt = runtime
        self.config = config
        self.executor = executor
        self.evaluator = evaluator

    @classmethod
    def load(cls, config: WorkerConfig, *, warmup: bool = True) -> "GpuWorker":
        """Load dots.tts and the executor once, then warm up (slow; do this at worker start)."""
        from adaptive_nfe.executor import Executor
        from dots_tts.runtime import DotsTtsRuntime

        log.info("loading %s (precision=%s, optimize=%s)", config.model, config.precision, config.optimize)
        rt = DotsTtsRuntime.from_pretrained(config.model, precision=config.precision, optimize=config.optimize)
        worker = cls(rt, config, Executor(rt.model), evaluator=build_evaluator(config))
        if warmup:
            worker.warmup()
        return worker

    def _generate(self, text: str, nfe: int):
        """generate_fixed_nfe at the worker's settings. Returns (result, ode_method, guidance_scale)."""
        c = self.config
        ode, steps, cfg = self.rt.resolve_sampling_options(ode_method=None, num_steps=nfe,
                                                           guidance_scale=c.guidance_scale)
        seed_all(c.seed)
        result = generate_fixed_nfe(self.rt, steps, executor=self.executor, text=text, prompt_audio_path=None,
                                    prompt_text=None, ode_method=ode, guidance_scale=cfg,
                                    speaker_scale=c.speaker_scale)
        return result, ode, cfg

    def warmup(self) -> None:
        """One discarded run per configured NFE (setup cost, and with optimize=True the first run at a
        new NFE rebuilds the KV cache). Nothing is stored."""
        for nfe in self.config.warmup_nfes:
            log.info("warmup nfe=%d (discarded)", nfe)
            self._generate(self.config.warmup_text, nfe)

    def run_job(self, job: Job) -> JobOutcome:
        """Generate, measure, validate and store one run. Never raises for a failed run."""
        try:
            return self._run_job(job)
        except Exception as e:  # noqa: BLE001  one bad run must not stop the worker
            log.exception("run %s failed", job.run_id)
            if isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()  # give the next run a clean slate
                return _failed("out of GPU memory")
            return _failed(f"{type(e).__name__}: {e}")

    def _run_job(self, job: Job) -> JobOutcome:
        c = self.config
        if isinstance(job.nfe, bool) or not isinstance(job.nfe, int) or job.nfe < 1:
            return _failed(f"invalid nfe: {job.nfe!r}")
        if not job.prompt or not job.prompt.strip():
            return _failed("empty prompt")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        # Raises if any patch missed the NFE or skipped the executor.
        gen, ode, cfg = self._generate(job.prompt, job.nfe)
        out = gen.output
        peak_vram_mb = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
        audio_t, sr = out["audio"], int(out["sample_rate"])

        result: dict[str, Any] = dict(
            nfe=job.nfe, nfe_per_patch=gen.nfe_per_patch, num_patches=gen.num_patches,
            executor_used_all_patches=gen.executor_nfe_per_patch == gen.nfe_per_patch,
            sample_rate=sr,
            # Runtime numbers (generation only), under "metrics" as agreed with Sai.
            metrics=dict(
                audio_s=round(audio_t.shape[-1] / sr, 4),
                time_s=round(float(out["time_used"]), 4), rtf=round(float(out["rtf"]), 6),
                peak_vram_mb=None if peak_vram_mb is None else round(peak_vram_mb, 1),
            ),
            audio_sha256=audio_sha256(audio_t),
            model=c.model, precision=c.precision, optimize=c.optimize, seed=c.seed, ode_method=ode,
            guidance_scale=cfg, speaker_scale=c.speaker_scale,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            worker_version=__version__,
            telemetry=None,  # runtime telemetry (Niranjan)
            quality=None,  # filled below once the clip has passed validation
        )

        # Check first, then store: the WAV is written only by validate_and_store, to
        # <storage>/<run_id>/ if it passed or _rejected/<run_id>/ if not.
        audio = audio_t.float().cpu().squeeze().numpy()
        validation = validate_and_store(audio, sr, job.run_id, c.storage_dir, num_patches=gen.num_patches,
                                        max_generate_length=self.rt.max_generate_length)
        result["validation"] = validation
        log.info("run %s nfe=%d: %d patches, %.2fs audio, RTF %.3f, validation %s", job.run_id, job.nfe,
                 gen.num_patches, result["metrics"]["audio_s"], result["metrics"]["rtf"],
                 "ok" if validation["ok"] else validation["flags"])
        if not validation["ok"]:
            return _failed("WAV failed validation: " + ", ".join(validation["flags"]), result)
        # Scored here, before the run is marked completed. evaluate() never raises: a scorer that fails
        # leaves its metric None and an entry in quality["errors"], it does not fail the run.
        if self.evaluator is not None:
            result["quality"] = self.evaluator.evaluate(validation["audio_path"], job.prompt, validation=validation)
        return JobOutcome(status="completed", result=result, audio_path=validation["audio_path"],
                          error_message=None)
