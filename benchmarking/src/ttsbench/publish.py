"""Publish a stored run: upload its directory to object storage, then index it in Postgres.

Artifacts go first so the index never points at files that don't exist; both steps are
idempotent, so re-publishing a run (e.g. after a partial failure) is safe.

Configured by environment:
    TTSBENCH_ARTIFACTS_URI   s3://bucket/prefix   (S3-compatible; set AWS_ENDPOINT_URL for R2/MinIO)
    TTSBENCH_DATABASE_URL    postgresql://...
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from ttsbench.storage import Run
from ttsbench.summary import flatten_row

ARTIFACTS_ENV = "TTSBENCH_ARTIFACTS_URI"
DATABASE_ENV = "TTSBENCH_DATABASE_URL"


class ArtifactStore(Protocol):
    def put(self, local: Path, key: str) -> None: ...

    def uri(self, key: str) -> str: ...


@dataclass(frozen=True)
class S3Store:
    bucket: str
    prefix: str = ""
    client: Any = None

    @classmethod
    def from_uri(cls, uri: str) -> S3Store:
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError(f"{ARTIFACTS_ENV} must look like s3://bucket/prefix, got {uri!r}")
        import boto3

        return cls(bucket=parsed.netloc, prefix=parsed.path.strip("/"), client=boto3.client("s3"))

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, local: Path, key: str) -> None:
        self.client.upload_file(str(local), self.bucket, self._key(key))

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{self._key(key)}"


def upload_run(run: Run, store: ArtifactStore) -> str:
    run_id = run.path.name
    for path in sorted(p for p in run.path.rglob("*") if p.is_file() and not p.name.endswith(".tmp")):
        store.put(path, f"{run_id}/{path.relative_to(run.path).as_posix()}")
    return store.uri(run_id)


def schema_sql() -> str:
    return resources.files("ttsbench").joinpath("sql/schema.sql").read_text()


def _run_row(run: Run, artifact_uri: str) -> dict[str, Any]:
    m = run.manifest
    return {
        "run_id": m["run_id"],
        "name": m["config"]["name"],
        "status": m["status"],
        "model": m["config"]["model"]["name"],
        "backend": m["backend"]["name"],
        "gpu": m["backend"].get("gpu"),
        "nfe": m["config"]["sweep"]["nfe"],
        "config_hash": m["config_hash"],
        "prompts_hash": m["prompts_hash"],
        "protocol_version": m["measurement_protocol_version"],
        "git_sha": m["git"].get("sha"),
        "git_dirty": m["git"].get("dirty"),
        "started_at": m["started_at"],
        "finished_at": m["finished_at"],
        "artifact_uri": artifact_uri,
        "manifest": json.dumps(m),
    }


def _summary_rows(run: Run, run_id: str) -> list[dict[str, Any]]:
    rows = []
    for row in run.summary():
        flat = flatten_row(row)
        rows.append(
            {
                **flat,
                "run_id": run_id,
                "device_p50_ms": flat["device_p50_ms"] if flat["device_p50_ms"] != "" else None,
                "stats": json.dumps(row),
            }
        )
    return rows


def _sample_rows(run: Run, run_id: str, artifact_uri: str) -> list[dict[str, Any]]:
    return [
        {
            **s.model_dump(
                include={
                    "prompt_id",
                    "nfe",
                    "repeat",
                    "seed",
                    "latency_ms",
                    "device_ms",
                    "to_host_ms",
                    "audio_seconds",
                    "rtf",
                    "audio_sha256",
                }
            ),
            "run_id": run_id,
            "audio_uri": f"{artifact_uri}/{s.audio_path}" if s.audio_path else None,
            "stages_ms": json.dumps(s.stages_ms),
            "telemetry": json.dumps(s.telemetry),
        }
        for s in run.samples()
    ]


def _insert(cur: Any, table: str, rows: list[dict[str, Any]], conflict: str = "") -> None:
    if not rows:
        return
    cols = list(rows[0])
    placeholders = ", ".join(f"%({c})s" for c in cols)
    cur.executemany(f"insert into {table} ({', '.join(cols)}) values ({placeholders}) {conflict}", rows)


def index_run(run: Run, artifact_uri: str, dsn: str) -> None:
    import psycopg

    run_row = _run_row(run, artifact_uri)
    run_id = run_row["run_id"]
    updates = ", ".join(f"{c} = excluded.{c}" for c in run_row if c != "run_id")
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(schema_sql())
        _insert(cur, "runs", [run_row], f"on conflict (run_id) do update set {updates}, published_at = now()")
        cur.execute("delete from run_summaries where run_id = %s", (run_id,))
        cur.execute("delete from samples where run_id = %s", (run_id,))
        _insert(cur, "run_summaries", _summary_rows(run, run_id))
        _insert(cur, "samples", _sample_rows(run, run_id, artifact_uri))


def publish_run(
    run: Run, store: ArtifactStore | None = None, dsn: str | None = None, allow_incomplete: bool = False
) -> str:
    status = run.manifest["status"]
    if status != "completed" and not allow_incomplete:
        raise ValueError(f"run {run.path.name} is {status!r}; pass allow_incomplete to publish it anyway")
    store = store or S3Store.from_uri(_require_env(ARTIFACTS_ENV))
    dsn = dsn or _require_env(DATABASE_ENV)
    artifact_uri = upload_run(run, store)
    index_run(run, artifact_uri, dsn)
    return artifact_uri


def check_publish_config() -> None:
    """Fail fast (before a long run) if publishing isn't configured."""
    for name in (ARTIFACTS_ENV, DATABASE_ENV):
        _require_env(name)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set")
    return value
