"""Correct wall-time measurement for model runs, on CPU or CUDA."""

import contextlib
import statistics
import time
from functools import partial

import torch

from .toy_model import run_toy_model


def _time_once_cpu(fn):
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000.0


def _time_once_cuda(fn):
    # Drain work queued before us so it isn't billed to this run.
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    # Events are filled in by the GPU; wait until it has reached `end`.
    end.synchronize()
    return start.elapsed_time(end)  # already in milliseconds


def time_callable(fn, steps, warmups=3, repeats=10, measure=None, min_duration_s=0.0):
    """Time `fn()` and return a metrics dict ready for a run-record.

    `steps` is not passed to `fn`; it is only used to compute per-step time.
    `measure`, if given, is a context manager entered around the timed repeats
    only (not the warm-ups), e.g. a GpuSampler, so it sees steady-state runs.
    Runs at least `repeats` times, and keeps going until the timed section has
    lasted `min_duration_s` of wall time; `repeats` in the result is the count
    that actually ran.
    """
    if steps < 1 or warmups < 0 or repeats < 1 or min_duration_s < 0:
        raise ValueError("need steps >= 1, warmups >= 0, repeats >= 1, min_duration_s >= 0")

    use_cuda = torch.cuda.is_available()
    time_once = _time_once_cuda if use_cuda else _time_once_cpu

    for _ in range(warmups):
        fn()

    samples_ms = []
    with measure or contextlib.nullcontext():
        # Clock starts inside `measure`, so sampler start-up isn't counted.
        start = time.perf_counter()
        while len(samples_ms) < repeats or time.perf_counter() - start < min_duration_s:
            samples_ms.append(time_once(fn))
        measured_s = time.perf_counter() - start
    # Every sample is one complete run of the same work, so the median is still
    # "time per run" however many there are; more repeats only tighten it.
    median_ms = statistics.median(samples_ms)

    return {
        "steps": steps,
        "median_ms": median_ms,
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "per_step_ms": median_ms / steps,
        "device": "cuda" if use_cuda else "cpu",
        "timer": "cuda_event" if use_cuda else "perf_counter",
        "warmups": warmups,
        "repeats": len(samples_ms),
        "min_repeats": repeats,
        "min_duration_s": float(min_duration_s),
        # Wall time of the whole timed section, including gaps between runs.
        # Not median_ms * repeats; it is the window the GPU sampler saw.
        "measured_s": measured_s,
    }


if __name__ == "__main__":
    for steps in (8, 32):
        r = time_callable(partial(run_toy_model, steps), steps)
        print(
            f"steps={r['steps']:>3}  median={r['median_ms']:8.2f} ms  "
            f"[min {r['min_ms']:.2f}, max {r['max_ms']:.2f}]  "
            f"per_step={r['per_step_ms']:.3f} ms  ({r['device']}, {r['timer']})"
        )
