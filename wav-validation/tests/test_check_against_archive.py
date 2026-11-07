"""The archive checker itself, on a tiny fake archive written by the old script's rules."""

import csv
import importlib.util
from pathlib import Path

from synth import clean, join, silence, tone, write

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_against_archive.py"
spec = importlib.util.spec_from_file_location("check_against_archive", SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

OLD = Path(__file__).resolve().parents[1] / "flag_wav_runs.py"
old_spec = importlib.util.spec_from_file_location("flag_wav_runs", OLD)
old = importlib.util.module_from_spec(old_spec)
old_spec.loader.exec_module(old)


def old_row(archive: Path, rel: str, max_generate_length: int = 80) -> dict:
    """A wav_flags.csv row computed exactly the way flag_wav_runs.py computes one."""
    c = old.signal_checks(archive / rel)
    cap = max_generate_length * old.PATCH_SECONDS
    return {
        "local_path": rel,
        "max_generate_length": str(max_generate_length),
        "duration_seconds": str(round(c["duration_seconds"], 4)),
        "peak": str(round(c["peak"], 4)),
        "clipped_samples": str(c["clipped_samples"]),
        "leading_silence_seconds": str(round(c["leading_silence_seconds"], 2)),
        "trailing_silence_seconds": str(round(c["trailing_silence_seconds"], 2)),
        "longest_internal_gap_seconds": str(round(c["longest_internal_gap_seconds"], 2)),
        "at_cap": str(c["duration_seconds"] >= cap - old.CAP_TOLERANCE_SECONDS),
        "signal_ok": str(old.signal_ok(c)),
    }


def write_flags(path: Path, rows: list) -> Path:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def test_agrees_with_old_script_on_varied_clips(tmp_path, capsys):
    clips = {
        "a/clean.wav": clean(),
        "a/abrupt.wav": join(silence(0.1), tone(1.0)),
        "a/gap.wav": join(silence(0.6), tone(0.4, amp=0.97), silence(1.3), tone(0.5), silence(0.2)),
        "b/capped.wav": join(silence(0.1), tone(12.7)),
    }
    for rel, audio in clips.items():
        write(tmp_path / rel, audio)
    flags = write_flags(tmp_path / "wav_flags.csv", [old_row(tmp_path, rel) for rel in clips])
    assert checker.main(["--archive", str(tmp_path), "--flags", str(flags)]) == 0
    assert "all agree" in capsys.readouterr().out


def test_reports_a_disagreement(tmp_path, capsys):
    write(tmp_path / "x.wav", clean())
    row = old_row(tmp_path, "x.wav")
    row["signal_ok"] = "False"  # pretend the old script said otherwise
    flags = write_flags(tmp_path / "wav_flags.csv", [row])
    assert checker.main(["--archive", str(tmp_path), "--flags", str(flags)]) == 1
    assert "signal_ok: 1 mismatches" in capsys.readouterr().out


def test_reports_missing_wavs(tmp_path, capsys):
    write(tmp_path / "x.wav", clean())
    row = old_row(tmp_path, "x.wav")
    (tmp_path / "x.wav").unlink()
    flags = write_flags(tmp_path / "wav_flags.csv", [row])
    assert checker.main(["--archive", str(tmp_path), "--flags", str(flags)]) == 1
    assert "missing_audio" in capsys.readouterr().out
