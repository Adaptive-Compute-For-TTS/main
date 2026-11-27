"""Generate one utterance with any Scheduler: each patch can run at a different NFE.

    torch.manual_seed(42)
    result = generate_scheduled(rt, ListScheduler([32, 8, 8, 16]), text="Hello there.", guidance_scale=1.2)
    result.nfe_per_patch      # what each patch actually ran at
    result.patch_ms           # per-patch DiT time (time_patches=True)

Works on both paths. On the optimized path (optimize=True), the DiT KV cache keeps one K/V copy per
ODE step, so a patch whose NFE differs from the previous patch's throws the cache away and recomputes
K/V for the whole history at every ODE step (dots.tts DiTSolver._ensure_cache + runner.prefill). That is
correct, just slow; it grows with the patch position. The eager path keeps no cache, so switching is free.

generate_fixed_nfe is this with a FixedScheduler plus a check that every patch got the one NFE.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch

from adaptive_nfe.scheduler import Scheduler

log = logging.getLogger(__name__)


@dataclass
class ScheduledResult:
    output: dict  # what runtime.generate returned (audio, sample_rate, time_used, rtf, ...)
    nfe_per_patch: list[int]  # from the scheduler, one entry per patch
    executor_nfe_per_patch: list[int] | None  # from the executor; None when it wasn't used
    patch_ms: list[float] = field(default_factory=list)  # DiT time per patch, if time_patches

    @property
    def num_patches(self) -> int:
        return len(self.nfe_per_patch)

    @property
    def nfe_switches(self) -> int:
        """How many times consecutive patches changed NFE (each one is a KV rebuild when optimized)."""
        return sum(a != b for a, b in zip(self.nfe_per_patch, self.nfe_per_patch[1:]))

    @property
    def total_dit_steps(self) -> int:
        return sum(self.nfe_per_patch)


class _RecordingExecutor:
    """Passes decode_patch through to the real executor, records each call's nfe (and time)."""

    def __init__(self, executor, time_patches: bool) -> None:
        self.executor = executor
        self.time_patches = time_patches
        self.calls: list[int] = []
        self.ms: list[float] = []

    def _run(self, fn, kwargs):
        self.calls.append(kwargs["nfe"])
        if not self.time_patches:
            return fn(**kwargs)
        sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
        sync()
        start = time.perf_counter()
        out = fn(**kwargs)
        sync()
        self.ms.append((time.perf_counter() - start) * 1e3)
        return out

    def decode_patch(self, **kwargs):
        return self._run(self.executor.decode_patch, kwargs)

    def decode_patch_optimized(self, **kwargs):
        return self._run(self.executor.decode_patch_optimized, kwargs)


def generate_scheduled(runtime, scheduler: Scheduler, *, use_executor: bool = True, executor=None,
                       time_patches: bool = False, **generate_kwargs) -> ScheduledResult:
    """Run runtime.generate(**generate_kwargs) with `scheduler` choosing every patch's NFE.

    `num_steps` (dots.tts's default NFE, which the scheduler overrides) defaults to 32 when not given.
    `executor`: reuse an adaptive_nfe.Executor across calls; one is built when None.
    `time_patches`: CUDA-synchronized DiT time per patch (adds a sync per patch; for measurement runs).
    """
    generate_kwargs.setdefault("num_steps", 32)
    model = runtime.model
    scheduler.reset()
    recorder = None
    if use_executor:
        if executor is None:
            from adaptive_nfe.executor import Executor  # imports dots.tts, so only when needed

            executor = Executor(model)
        if getattr(model, "_optimize_enabled", False) and not hasattr(executor, "decode_patch_optimized"):
            raise NotImplementedError("This executor has no decode_patch_optimized (optimize=True); "
                                      "pass use_executor=False.")
        recorder = _RecordingExecutor(executor, time_patches)

    saved = (model.nfe_scheduler, model.dit_executor)
    model.nfe_scheduler, model.dit_executor = scheduler, recorder
    try:
        output = runtime.generate(**generate_kwargs)
    finally:
        model.nfe_scheduler, model.dit_executor = saved

    nfe_per_patch = [h["nfe"] for h in scheduler.history]
    if not nfe_per_patch:
        raise RuntimeError("No patches were decoded; is the dots.tts scheduler hook installed?")
    if recorder is not None and recorder.calls != nfe_per_patch:
        raise RuntimeError(f"Executor decoded {len(recorder.calls)} patches {recorder.calls}, "
                           f"scheduler saw {len(nfe_per_patch)} {nfe_per_patch}.")
    log.info("scheduled: %d patches, nfe %s", len(nfe_per_patch), nfe_per_patch)
    return ScheduledResult(output, nfe_per_patch, recorder.calls if recorder else None,
                           recorder.ms if recorder else [])
