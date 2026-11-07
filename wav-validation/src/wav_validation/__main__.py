"""Check WAV files from the command line (e.g. on the GPU box after a test generation).

    python -m wav_validation audio.wav other.wav

Prints OK or FAIL per file with its flags; exits 1 if any file fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from wav_validation.checks import validate_wav


def line(path: Path, result: dict) -> str:
    status = "OK  " if result["ok"] else "FAIL"
    flags = ", ".join(result["flags"]) or "-"
    return f"{status}  {path}  {flags}"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m wav_validation", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", type=Path, help="WAV files")
    a = p.parse_args(argv)

    failed = 0
    for path in a.paths:
        result = validate_wav(path)
        failed += not result["ok"]
        print(line(path, result))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
