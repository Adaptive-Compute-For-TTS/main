"""Capture the hardware/software context a measurement was taken in."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

_ENV_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_LAUNCH_BLOCKING",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TORCH_CUDNN_V8_API_ENABLED",
    "OMP_NUM_THREADS",
    "MODAL_TASK_ID",
    "MODAL_REGION",
    "MODAL_CLOUD_PROVIDER",
)

_PACKAGES = ("ttsbench", "torch", "triton", "numpy", "dots_tts", "dots.tts", "flash_attn", "modal")


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {}
    info: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            gpu_count=torch.cuda.device_count(),
            gpu_name=props.name,
            gpu_memory_gb=round(props.total_memory / 1024**3, 2),
            gpu_sm=f"{props.major}.{props.minor}",
            gpu_multiprocessors=props.multi_processor_count,
        )
    return info


def _nvidia_smi() -> dict[str, Any]:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,clocks.max.sm,clocks.max.mem,power.limit,pstate",
            "--format=csv,noheader",
        ]
    )
    if not out:
        return {}
    first = [part.strip() for part in out.splitlines()[0].split(",")]
    keys = ("driver_version", "max_sm_clock", "max_mem_clock", "power_limit", "pstate")
    return dict(zip(keys, first, strict=False))


def worker_environment() -> dict[str, Any]:
    """Collected inside the worker, where the GPU is.

    Round-tripped through JSON so only builtin types leave the worker: e.g. `torch.__version__`
    is a `TorchVersion`, which a client without torch cannot unpickle.
    """
    env = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": _package_versions(),
        "torch": _torch_info(),
        "nvidia_smi": _nvidia_smi(),
        "env": {k: os.environ[k] for k in _ENV_VARS if k in os.environ},
    }
    return json.loads(json.dumps(env, default=str))


def git_state(repo: Path) -> dict[str, Any]:
    """Collected on the client, where the source tree is."""
    sha = _run(["git", "-C", str(repo), "rev-parse", "HEAD"])
    status = _run(["git", "-C", str(repo), "status", "--porcelain"])
    return {"sha": sha, "dirty": bool(status), "branch": _run(["git", "-C", str(repo), "branch", "--show-current"])}
