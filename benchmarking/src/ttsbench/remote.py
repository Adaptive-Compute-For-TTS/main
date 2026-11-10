"""File-based worker entrypoint for batch backends (SLURM): job.json in, result.json out.

python -m ttsbench.remote job.json result.json
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ttsbench.schema import Job
from ttsbench.worker import execute_job


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: python -m ttsbench.remote JOB_JSON RESULT_JSON", file=sys.stderr)
        return 2
    job = Job.model_validate_json(Path(argv[1]).read_text())
    result = execute_job(job)
    out = Path(argv[2])
    tmp = out.with_suffix(".tmp")
    tmp.write_text(result.model_dump_json())
    os.replace(tmp, out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
