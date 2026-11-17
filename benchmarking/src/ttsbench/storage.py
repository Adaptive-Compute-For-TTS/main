"""Filesystem run store. One directory per run, written incrementally so a crash mid-sweep
keeps every completed NFE value:

    runs/<run_id>/
      manifest.json    config, hashes, git, backend, protocol version, status, timings
      prompts.jsonl    exact prompts used
      workers.jsonl    worker environment + model load/warmup per job (one line per NFE)
      samples.jsonl    one line per measured synthesis
      audio/<prompt_id>/nfeNNN_rNN.wav
      summary.json / summary.csv
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ttsbench.schema import ExperimentConfig, JobResult, Prompt, Sample
from ttsbench.summary import flatten_row, summarize


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@dataclass(frozen=True)
class Run:
    path: Path

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads((self.path / "manifest.json").read_text())

    def prompts(self) -> list[Prompt]:
        return [Prompt.model_validate(row) for row in _read_jsonl(self.path / "prompts.jsonl")]

    def samples(self) -> list[Sample]:
        return [Sample.model_validate(row) for row in _read_jsonl(self.path / "samples.jsonl")]

    def workers(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.path / "workers.jsonl")

    def summary(self) -> list[dict[str, Any]]:
        path = self.path / "summary.json"
        return json.loads(path.read_text()) if path.exists() else self.write_summary()

    def write_summary(self) -> list[dict[str, Any]]:
        rows = summarize(self.samples(), self.prompts())
        _write_json_atomic(self.path / "summary.json", rows)
        flat = [flatten_row(r) for r in rows]
        if flat:
            with (self.path / "summary.csv").open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(flat[0]))
                writer.writeheader()
                writer.writerows(flat)
        return rows


class RunWriter:
    def __init__(self, run: Run, manifest: dict[str, Any]) -> None:
        self.run = run
        self._manifest = manifest

    @classmethod
    def create(
        cls,
        run_id: str,
        config: ExperimentConfig,
        prompts: list[Prompt],
        metadata: dict[str, Any],
    ) -> RunWriter:
        path = Path(config.output_dir) / run_id
        path.mkdir(parents=True, exist_ok=False)
        (path / "audio").mkdir()
        append_jsonl(path / "prompts.jsonl", [p.model_dump(mode="json") for p in prompts])
        manifest = {
            "run_id": run_id,
            "status": "running",
            "started_at": _now(),
            "finished_at": None,
            "config": config.model_dump(mode="json"),
            **metadata,
        }
        _write_json_atomic(path / "manifest.json", manifest)
        return cls(Run(path), manifest)

    def record(self, result: JobResult) -> None:
        for key, data in result.audio.items():
            target = self.run.path / "audio" / f"{key}.wav"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        append_jsonl(self.run.path / "workers.jsonl", [{"nfe": result.nfe, **result.worker.model_dump()}])
        append_jsonl(self.run.path / "samples.jsonl", [s.model_dump(mode="json") for s in result.samples])

    def finish(self, status: str, error: str | None = None) -> None:
        self._manifest = {**self._manifest, "status": status, "finished_at": _now(), "error": error}
        _write_json_atomic(self.run.path / "manifest.json", self._manifest)
        if self.run.samples():
            self.run.write_summary()


def list_runs(root: Path) -> list[Run]:
    if not root.is_dir():
        return []
    return sorted(
        (Run(p) for p in root.iterdir() if (p / "manifest.json").is_file()),
        key=lambda r: r.path.name,
    )
