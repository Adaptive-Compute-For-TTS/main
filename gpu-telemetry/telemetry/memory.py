"""Measure peak GPU memory for a single run."""

from pprint import pprint

import torch

from .fingerprint import NOT_AVAILABLE
from .toy_model import run_toy_model

BYTES_PER_MB = 2**20


def reset():
    """Zero the peak counters so the next read covers only what runs after this."""
    if not torch.cuda.is_available():
        return
    torch.cuda.reset_peak_memory_stats()


def read_peak():
    """Return peak allocated/reserved memory in MB since the last reset().

    Works on CPU-only machines: values are filled with placeholders.
    """
    if not torch.cuda.is_available():
        return {"peak_allocated_mb": NOT_AVAILABLE, "peak_reserved_mb": NOT_AVAILABLE}

    return {
        # Bytes held by live tensors at their high-water mark.
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / BYTES_PER_MB,
        # Bytes the caching allocator had claimed from the driver at its high-water mark.
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / BYTES_PER_MB,
    }


if __name__ == "__main__":
    reset()
    run_toy_model(32)
    pprint(read_peak(), sort_dicts=False)
