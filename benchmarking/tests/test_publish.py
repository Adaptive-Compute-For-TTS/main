from __future__ import annotations

import os
from pathlib import Path

import pytest

from ttsbench.publish import (
    ARTIFACTS_ENV,
    S3Store,
    _run_row,
    _sample_rows,
    _summary_rows,
    check_publish_config,
    index_run,
    publish_run,
    upload_run,
)
from ttsbench.runner import run_experiment
from ttsbench.schema import ExperimentConfig


class FakeStore:
    def __init__(self) -> None:
        self.keys: list[str] = []

    def put(self, local: Path, key: str) -> None:
        assert local.is_file()
        self.keys.append(key)

    def uri(self, key: str) -> str:
        return f"s3://bucket/{key}"


class FakeS3Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.calls.append((filename, bucket, key))


def test_upload_run_uploads_every_file_under_run_id(config: ExperimentConfig) -> None:
    run = run_experiment(config)
    store = FakeStore()
    uri = upload_run(run, store)
    name = run.path.name
    assert uri == f"s3://bucket/{name}"
    assert {f"{name}/manifest.json", f"{name}/samples.jsonl", f"{name}/summary.json"} <= set(store.keys)
    assert sum(k.endswith(".wav") for k in store.keys) == 6


def test_s3_store_prefixing() -> None:
    client = FakeS3Client()
    store = S3Store(bucket="b", prefix="bench/runs", client=client)
    store.put(Path("x"), "r1/manifest.json")
    assert client.calls == [("x", "b", "bench/runs/r1/manifest.json")]
    assert store.uri("r1") == "s3://b/bench/runs/r1"
    with pytest.raises(ValueError):
        S3Store.from_uri("https://not-s3")


def test_rows_match_schema_columns(config: ExperimentConfig) -> None:
    run = run_experiment(config)
    schema = (Path(__file__).parents[1] / "src/ttsbench/sql/schema.sql").read_text()
    for rows in (
        [_run_row(run, "s3://b/r")],
        _summary_rows(run, "r"),
        _sample_rows(run, "r", "s3://b/r"),
    ):
        for column in rows[0]:
            assert f"    {column} " in schema, column
    samples = _sample_rows(run, "r", "s3://b/r")
    assert len(samples) == 18
    assert sum(r["audio_uri"] is not None for r in samples) == 6


def test_publish_refuses_incomplete_runs(config: ExperimentConfig) -> None:
    run = run_experiment(config)
    (run.path / "manifest.json").write_text((run.path / "manifest.json").read_text().replace('"completed"', '"failed"'))
    with pytest.raises(ValueError, match="failed"):
        publish_run(run, store=FakeStore(), dsn="postgresql://unused")


def test_check_publish_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ARTIFACTS_ENV, raising=False)
    with pytest.raises(RuntimeError, match=ARTIFACTS_ENV):
        check_publish_config()


@pytest.mark.skipif(not os.environ.get("TTSBENCH_TEST_DATABASE_URL"), reason="needs TTSBENCH_TEST_DATABASE_URL")
def test_index_run_is_idempotent(config: ExperimentConfig) -> None:
    import psycopg

    dsn = os.environ["TTSBENCH_TEST_DATABASE_URL"]
    run = run_experiment(config)
    index_run(run, "s3://b/r", dsn)
    index_run(run, "s3://b/r", dsn)
    with psycopg.connect(dsn) as conn:
        run_id = run.path.name
        count = lambda t: conn.execute(f"select count(*) from {t} where run_id = %s", (run_id,)).fetchone()[0]  # noqa: E731
        assert (count("runs"), count("run_summaries"), count("samples")) == (1, 8, 18)
        conn.execute("delete from runs where run_id = %s", (run_id,))
