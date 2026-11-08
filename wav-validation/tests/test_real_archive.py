"""Smoke test on real dots.tts output from the archived benchmark. Opt-in:

    WAV_VALIDATION_ARCHIVE=/path/to/watai_exp pytest tests/test_real_archive.py
"""

import csv
import os
from pathlib import Path

import pytest
import soundfile as sf

from wav_validation import validate_and_store

ARCHIVE = os.environ.get("WAV_VALIDATION_ARCHIVE")
FLAGS_CSV = Path(__file__).resolve().parents[1] / "wav_flags.csv"
pytestmark = pytest.mark.skipif(not ARCHIVE, reason="set WAV_VALIDATION_ARCHIVE to a watai_exp clone")


def first_row(status: str) -> dict:
    with FLAGS_CSV.open(newline="") as f:
        return next(r for r in csv.DictReader(f) if r["conditioning"] == "x-vector" and r["status"] == status)


def load(row: dict):
    return sf.read(Path(ARCHIVE).expanduser() / row["local_path"], dtype="float32")


def test_real_clean_clip_is_stored(tmp_path):
    audio, sr = load(first_row("clean_candidate"))
    r = validate_and_store(audio, sr, "clean", tmp_path, max_generate_length=80)
    assert r["ok"] is True and r["audio_path"] == str((tmp_path / "clean" / "audio.wav").resolve())
    assert sr == 48_000


def test_real_cut_off_clip_is_rejected(tmp_path):
    audio, sr = load(first_row("truncated_at_cap"))
    r = validate_and_store(audio, sr, "capped", tmp_path, max_generate_length=80)
    assert r["ok"] is False and "truncated_at_cap" in r["flags"]
    assert r["audio_path"] is None and "_rejected" in r["stored_path"]
