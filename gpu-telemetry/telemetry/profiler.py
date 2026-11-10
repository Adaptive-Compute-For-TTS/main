"""Count kernel launches and per-op costs for one run, using torch.profiler.

This is a characterization pass, not a timing pass: the profiler slows every
op down, so call it on its own, never inside time_callable()'s timed loop.
"""

from functools import partial
from pprint import pprint

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from .fingerprint import NOT_AVAILABLE
from .toy_model import run_toy_model

DEFAULT_TOP_N = 5
US_PER_MS = 1000.0
# Device-side events that are copies/fills, not kernels.
NON_KERNEL_PREFIXES = ("Memcpy", "Memset")


def profile_callable(fn, steps, warmups=1, top_n=DEFAULT_TOP_N):
    """Profile one call of `fn()` and return a summary dict ready for a run-record.

    `steps` is not passed to `fn`; it is recorded so counts can be read per step.
    Works on CPU-only machines: CUDA-only fields are filled with placeholders.
    """
    if steps < 1 or warmups < 0 or top_n < 0:
        raise ValueError("need steps >= 1, warmups >= 0, top_n >= 0")

    use_cuda = torch.cuda.is_available()
    activities = [ProfilerActivity.CPU]
    if use_cuda:
        activities.append(ProfilerActivity.CUDA)

    # Outside the profiler, so one-time setup (cuBLAS init, lazy loading)
    # isn't counted as part of the run.
    for _ in range(warmups):
        fn()

    with profile(activities=activities) as prof:
        fn()
    events = prof.events()

    # Ops the Python code called directly. Ops they call internally
    # (aten::matmul -> aten::mm) have a cpu_parent and are left out.
    top_level_ops = [
        e for e in events
        if e.device_type == DeviceType.CPU and e.cpu_parent is None and e.name.startswith("aten::")
    ]
    kernels = [
        e for e in events
        if e.device_type == DeviceType.CUDA and not e.name.startswith(NON_KERNEL_PREFIXES)
    ]

    summary = {
        "steps": steps,
        "device": "cuda" if use_cuda else "cpu",
        "warmups": warmups,
        "cpu_op_calls": len(top_level_ops),
        "distinct_cpu_ops": len({e.name for e in top_level_ops}),
        "kernel_launches": NOT_AVAILABLE,
        "distinct_kernels": NOT_AVAILABLE,
        "gpu_kernel_time_ms": NOT_AVAILABLE,
        "top_ops": _top_ops(prof, use_cuda, top_n),
    }
    # On a GPU the model always launches kernels, so finding none means CUDA
    # tracing failed (e.g. CUPTI unavailable); "n/a" beats a misleading 0.
    if kernels:
        summary.update(
            kernel_launches=len(kernels),
            distinct_kernels=len({e.name for e in kernels}),
            gpu_kernel_time_ms=sum(e.time_range.elapsed_us() for e in kernels) / US_PER_MS,
        )
    return summary


def _top_ops(prof, use_cuda, top_n):
    """The `top_n` aten ops by self time: GPU time on CUDA, CPU time otherwise."""
    ops = [avg for avg in prof.key_averages() if avg.key.startswith("aten::")]
    # Self time is an op's own cost, excluding ops it called, so ranking by it
    # never counts the same work twice (aten::matmul vs the aten::mm inside it).
    if use_cuda:
        ops.sort(key=lambda avg: avg.self_device_time_total, reverse=True)
    else:
        ops.sort(key=lambda avg: avg.self_cpu_time_total, reverse=True)
    return [
        {
            "name": avg.key,
            "calls": avg.count,
            "self_cpu_time_ms": avg.self_cpu_time_total / US_PER_MS,
            "self_gpu_time_ms": (
                avg.self_device_time_total / US_PER_MS if use_cuda else NOT_AVAILABLE
            ),
        }
        for avg in ops[:top_n]
    ]


if __name__ == "__main__":
    steps = 32
    pprint(profile_callable(partial(run_toy_model, steps), steps), sort_dicts=False)
