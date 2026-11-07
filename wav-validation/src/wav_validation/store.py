"""Validation -> Local WAV Storage: save each clip under its run id, keeping failed clips apart.

Layout under storage_dir:
    <run_id>/audio.wav             passed: this is the run's audio_path, served to the frontend
    <run_id>/validation.json
    _rejected/<run_id>/audio.wav   failed: kept for debugging, never served
    _rejected/<run_id>/validation.json
"""

from __future__ import annotations

import re

REJECTED_DIR = "_rejected"

# Backend run ids are UUIDs. Anything that could escape storage_dir ("..", "a/b") or collide with
# REJECTED_DIR (leading "_" or ".") is refused.
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def check_run_id(run_id: str) -> str:
    """Return run_id if it is safe to use as a folder name, else raise ValueError."""
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id) or ".." in run_id:
        raise ValueError(f"unsafe run_id for a storage folder: {run_id!r}")
    return run_id
