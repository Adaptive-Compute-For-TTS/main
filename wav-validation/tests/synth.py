"""Synthetic audio for the tests: tones and silence at 48 kHz, the rate dots.tts outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SR = 48_000


def tone(seconds: float, amp: float = 0.5, freq: float = 220.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(round(seconds * sr))) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(round(seconds * sr)), dtype=np.float32)


def join(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts).astype(np.float32)


def clean(seconds: float = 1.0, sr: int = SR) -> np.ndarray:
    """A clip that passes every check: short lead-in, steady tone, short tail."""
    return join(silence(0.1, sr), tone(seconds, sr=sr), silence(0.1, sr))


def write(path: Path, audio: np.ndarray, sr: int = SR, subtype: str = "FLOAT") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, subtype=subtype)
    return path
