"""Check WAV files from the command line (e.g. on the GPU box after a test generation).

    python -m wav_validation audio.wav runs/

Directories are searched recursively for *.wav. Prints OK or FAIL per file with its flags and a
summary; exits 1 if any file fails.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

from wav_validation.checks import validate_wav


def line(path: Path, result: dict) -> str:
    status = "OK  " if result["ok"] else "FAIL"
    flags = ", ".join(result["flags"]) or "-"
    return f"{status}  {path}  {flags}"


def expand(paths: List[Path]) -> List[Path]:
    """Files as given; directories replaced by their *.wav files (recursive, sorted)."""
    out: List[Path] = []
    for path in paths:
        if path.is_dir():
            out += sorted(q for q in path.rglob("*") if q.suffix.lower() == ".wav" and q.is_file())
        else:
            out.append(path)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m wav_validation", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", type=Path, help="WAV files and/or directories")
    a = p.parse_args(argv)

    files = expand(a.paths)
    if not files:
        print("no .wav files found", file=sys.stderr)
        return 1
    failed = 0
    flag_counts: Counter = Counter()
    for path in files:
        result = validate_wav(path)
        failed += not result["ok"]
        flag_counts.update(result["flags"])
        print(line(path, result))
    summary = f"{len(files)} files: {len(files) - failed} ok, {failed} failed"
    if flag_counts:
        summary += " (" + ", ".join(f"{flag} {n}" for flag, n in flag_counts.most_common()) + ")"
    print(summary)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
