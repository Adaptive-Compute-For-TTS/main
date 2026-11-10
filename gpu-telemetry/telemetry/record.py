"""The run-record: everything about one model run, saved as one JSON file."""

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

from .fingerprint import get_fingerprint

SCHEMA_VERSION = 1
# Fixed US Eastern Standard Time (UTC-5), no daylight saving, so timestamps
# never jump or repeat. created_at carries the -05:00 offset explicitly.
EST = timezone(timedelta(hours=-5), "EST")
# Anchor outputs/ to the repo root so it doesn't depend on the current directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(REPO_ROOT, "outputs")


@dataclass
class RunRecord:
    run_id: str = field(init=False)
    created_at: str = field(init=False)
    schema_version: int = field(init=False, default=SCHEMA_VERSION)
    config: dict = field(default_factory=dict)
    fingerprint: dict = field(default_factory=get_fingerprint)
    metrics: dict = field(default_factory=dict)

    def __post_init__(self):
        # One clock read, so run_id and created_at always agree.
        now = datetime.now(EST)
        self.created_at = now.isoformat(timespec="seconds")
        self.run_id = f"{now:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:4]}"

    def save(self, out_dir=OUTPUTS_DIR):
        """Write the record to <out_dir>/run_<run_id>.json and return the path.

        Safe to call repeatedly (e.g. after filling in metrics): the file is
        replaced atomically, so a crash never leaves half-written JSON.
        """
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"run_{self.run_id}.json")
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_path, path)
        return path


if __name__ == "__main__":
    record = RunRecord(
        config={
            "nfe": 16,
            "utterance_id": "sample_0001",
            "text": "The quick brown fox jumps over the lazy dog.",
        }
    )
    print(record.save())
