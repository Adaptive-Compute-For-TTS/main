"""Check that wav_validation agrees with the old flag_wav_runs.py on the archived benchmark WAVs.

For every row of wav_flags.csv, runs validate_wav() on the same WAV (with the row's
max_generate_length, so the cut-off check uses the duration fallback like the old script) and
compares duration, peak, clipped samples, leading/trailing silence, longest gap, at_cap and
signal_ok at the precision the CSV stores. Exits 1 on any mismatch.

    python scripts/check_against_archive.py --archive ~/Desktop/work/watai/dots.tts

--archive is a clone of watai_exp: the folder that holds experiments/nfe_benchmark_runs/.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from wav_validation import validate_wav

HERE = Path(__file__).resolve().parent
DEFAULT_FLAGS = HERE.parent / "wav_flags.csv"

# wav_flags.csv column -> (validate_wav checks key, digits the CSV was rounded to; None = exact)
NUMERIC = {
    "duration_seconds": ("duration_s", 4),
    "peak": ("peak", 4),
    "clipped_samples": ("clipped_samples", None),
    "leading_silence_seconds": ("leading_silence_s", 2),
    "trailing_silence_seconds": ("trailing_silence_s", 2),
    "longest_internal_gap_seconds": ("longest_gap_s", 2),
}


def compare_row(row: dict, result: dict) -> list:
    """Names of the columns where the new result disagrees with the old CSV row."""
    bad = []
    checks = result["checks"]
    for column, (key, digits) in NUMERIC.items():
        new = checks[key]
        old = float(row[column])
        if new is None or (round(new, digits) if digits is not None else new) != old:
            bad.append(column)
    if str(checks["at_cap"]) != row["at_cap"]:
        bad.append("at_cap")
    if str(result["signal_ok"]) != row["signal_ok"]:
        bad.append("signal_ok")
    return bad


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archive", type=Path, required=True, help="watai_exp clone (has experiments/)")
    p.add_argument("--flags", type=Path, default=DEFAULT_FLAGS, help="wav_flags.csv to compare against")
    p.add_argument("--show", type=int, default=10, help="mismatching rows to print")
    a = p.parse_args(argv)

    archive = a.archive.expanduser().resolve()
    with a.flags.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    mismatches = Counter()
    shown = 0
    for row in rows:
        result = validate_wav(archive / row["local_path"], max_generate_length=int(row["max_generate_length"]))
        if result["flags"][:1] in (["missing_audio"], ["unreadable_audio"]):
            bad = [result["flags"][0]]
        else:
            bad = compare_row(row, result)
        mismatches.update(bad)
        if bad and shown < a.show:
            shown += 1
            print(f"MISMATCH {row['local_path']}: {', '.join(bad)}")

    print(f"{len(rows)} rows compared")
    if not mismatches:
        print("all agree")
        return 0
    for column, count in mismatches.most_common():
        print(f"  {column}: {count} mismatches")
    return 1


if __name__ == "__main__":
    sys.exit(main())
