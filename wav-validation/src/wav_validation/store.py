"""Validation -> Local WAV Storage: save each clip under its run id, keeping failed clips apart.

Layout under storage_dir:
    <run_id>/audio.wav             passed: this is the run's audio_path, served to the frontend
    <run_id>/validation.json
    _rejected/<run_id>/audio.wav   failed: kept for debugging, never served
    _rejected/<run_id>/validation.json
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import soundfile as sf

from wav_validation.checks import DEFAULT, Thresholds, validate_audio

REJECTED_DIR = "_rejected"

# Backend run ids are UUIDs. Anything that could escape storage_dir ("..", "a/b") or collide with
# REJECTED_DIR (leading "_" or ".") is refused.
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def check_run_id(run_id: str) -> str:
    """Return run_id if it is safe to use as a folder name, else raise ValueError."""
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id) or ".." in run_id:
        raise ValueError(f"unsafe run_id for a storage folder: {run_id!r}")
    return run_id


def write_wav_atomic(path: Path, audio: Any, sample_rate: int) -> Path:
    """Write float32 WAV (same format as gpu-executor/scripts/generate.py) via a temp file + rename,
    so a reader polling for the file (FastAPI results) never sees a half-written WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(audio, dtype=np.float32)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        sf.write(tmp, samples, int(sample_rate), subtype="FLOAT", format="WAV")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


def validate_and_store(audio: Any, sample_rate: int, run_id: str, storage_dir: Union[str, Path], *,
                       num_patches: Optional[int] = None, max_generate_length: Optional[int] = None,
                       thresholds: Thresholds = DEFAULT) -> Dict[str, Any]:
    """Validate a freshly generated clip and save it to Local WAV Storage.

    Returns validate_audio()'s result plus:
      audio_path   absolute path of the stored WAV if the clip passed (the run's audio_path), else None
      stored_path  where the WAV was written
    """
    check_run_id(run_id)
    result = validate_audio(audio, sample_rate, num_patches=num_patches,
                            max_generate_length=max_generate_length, thresholds=thresholds)
    storage = Path(storage_dir).expanduser().resolve()
    if not result["ok"]:
        return {**result, "audio_path": None, "stored_path": None}
    wav = write_wav_atomic(storage / run_id / "audio.wav", audio, sample_rate)
    return {**result, "audio_path": str(wav), "stored_path": str(wav)}
