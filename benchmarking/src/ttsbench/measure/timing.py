"""Latency measurement primitives.

Rules this module enforces (see README "How latency is measured"):

1. The device is synchronised immediately before the clock starts and immediately after the
   model returns, so queued GPU work is neither leaked into nor dropped from a measurement.
2. Wall time uses `time.perf_counter_ns` (monotonic, highest resolution available).
3. On CUDA, a pair of CUDA events brackets the same region to give device-side time.
4. Stage timings use CUDA events on CUDA and never force a host sync, so instrumenting stages
   does not change the latency being measured.
5. GC is optionally disabled inside the timed region.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")

_NS_PER_MS = 1_000_000


def _torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


class Device:
    """Thin wrapper over the device a model runs on, so timing code is torch-optional."""

    def __init__(self, kind: str) -> None:
        if kind not in ("cpu", "cuda"):
            raise ValueError(f"unsupported device {kind!r}")
        self._torch = _torch() if kind == "cuda" else None
        if kind == "cuda" and (self._torch is None or not self._torch.cuda.is_available()):
            raise RuntimeError("model requested device 'cuda' but CUDA is not available")
        self.kind = kind

    @property
    def is_cuda(self) -> bool:
        return self.kind == "cuda"

    def synchronize(self) -> None:
        if self._torch is not None:
            self._torch.cuda.synchronize()

    def event(self) -> Any:
        assert self._torch is not None
        return self._torch.cuda.Event(enable_timing=True)


@dataclass
class StageTimer:
    """Handed to `TTSModel.synthesize` so adapters can attribute time to named stages.

    Repeated stages (e.g. one "flow_step" per NFE) are summed and counted.
    """

    device: Device
    _cuda: list[tuple[str, Any, Any]] = field(default_factory=list)
    _host: list[tuple[str, int]] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if self.device.is_cuda:
            start, end = self.device.event(), self.device.event()
            start.record()
            try:
                yield
            finally:
                end.record()
                self._cuda.append((name, start, end))
        else:
            t0 = time.perf_counter_ns()
            try:
                yield
            finally:
                self._host.append((name, time.perf_counter_ns() - t0))

    def resolve(self) -> tuple[dict[str, float], dict[str, int]]:
        """Call only after the device has been synchronised."""
        totals: dict[str, float] = {}
        calls: dict[str, int] = {}
        for name, start, end in self._cuda:
            totals[name] = totals.get(name, 0.0) + float(start.elapsed_time(end))
            calls[name] = calls.get(name, 0) + 1
        for name, ns in self._host:
            totals[name] = totals.get(name, 0.0) + ns / _NS_PER_MS
            calls[name] = calls.get(name, 0) + 1
        return totals, calls


@dataclass(frozen=True)
class Timing:
    latency_ms: float
    device_ms: float | None
    stages_ms: dict[str, float]
    stage_calls: dict[str, int]


def measure(fn: Callable[[StageTimer], T], device: Device, *, disable_gc: bool = True) -> tuple[T, Timing]:
    """Run `fn` once inside a synchronised timed region."""
    stages = StageTimer(device)
    gc_was_enabled = gc.isenabled()
    if disable_gc:
        gc.collect()
        gc.disable()
    try:
        device.synchronize()
        start_evt = end_evt = None
        if device.is_cuda:
            start_evt, end_evt = device.event(), device.event()
            start_evt.record()
        t0 = time.perf_counter_ns()
        result = fn(stages)
        if end_evt is not None:
            end_evt.record()
        device.synchronize()
        t1 = time.perf_counter_ns()
    finally:
        if disable_gc and gc_was_enabled:
            gc.enable()

    device_ms = float(start_evt.elapsed_time(end_evt)) if start_evt is not None else None
    stages_ms, stage_calls = stages.resolve()
    return result, Timing(
        latency_ms=(t1 - t0) / _NS_PER_MS,
        device_ms=device_ms,
        stages_ms=stages_ms,
        stage_calls=stage_calls,
    )


def host_timed(fn: Callable[[], T]) -> tuple[T, float]:
    """Plain wall-clock timing for host-only work (no device sync)."""
    t0 = time.perf_counter_ns()
    result = fn()
    return result, (time.perf_counter_ns() - t0) / _NS_PER_MS
