"""Audio conversion helpers. Audio is float32 mono everywhere inside ttsbench."""

from __future__ import annotations

import hashlib
import io
from typing import Any

import numpy as np
import soundfile as sf


def to_numpy(audio: Any) -> np.ndarray:
    """Accept numpy arrays or torch tensors (any device/dtype) and return 1-D float32."""
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    array = np.asarray(audio, dtype=np.float32).squeeze()
    if array.ndim != 1:
        raise ValueError(f"expected mono audio, got shape {array.shape}")
    return array


def fingerprint(audio: np.ndarray) -> str:
    """SHA-256 of the raw float32 samples: identical iff the waveforms are bit-identical."""
    return hashlib.sha256(np.ascontiguousarray(audio, dtype=np.float32).tobytes()).hexdigest()


def encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(io.BytesIO(data), dtype="float32")
    return audio, sample_rate
