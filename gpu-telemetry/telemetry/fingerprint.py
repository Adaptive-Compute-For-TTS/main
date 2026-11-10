"""Capture the GPU/runtime environment a telemetry run executes on."""

import platform
from pprint import pprint

import torch

NOT_AVAILABLE = "n/a"


def _driver_version():
    """NVIDIA driver version, read from the driver itself via NVML."""
    try:
        import pynvml
    except ImportError:
        return NOT_AVAILABLE

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        return NOT_AVAILABLE

    try:
        return pynvml.nvmlSystemGetDriverVersion()
    except pynvml.NVMLError:
        return NOT_AVAILABLE
    finally:
        pynvml.nvmlShutdown()


def get_fingerprint(index=0):
    """Return a dict describing the GPU and software stack.

    Works on CPU-only machines: GPU fields are filled with placeholders.
    """
    fp = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        # CUDA version torch was *built* against; None for CPU-only builds.
        "torch_cuda_version": torch.version.cuda or NOT_AVAILABLE,
        "device": "cpu",
        "gpu_name": NOT_AVAILABLE,
        "gpu_total_memory_mb": NOT_AVAILABLE,
        "driver_version": NOT_AVAILABLE,
    }

    if not torch.cuda.is_available():
        return fp

    props = torch.cuda.get_device_properties(index)
    fp.update(
        device=f"cuda:{index}",
        gpu_name=props.name,
        gpu_total_memory_mb=props.total_memory // 2**20,
        driver_version=_driver_version(),
    )
    return fp


if __name__ == "__main__":
    pprint(get_fingerprint(), sort_dicts=False)
