import time
from typing import Any, Dict
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import func, select

from backend.app import repository
from backend.app.database import SessionLocal
from backend.app.main import create_app
from backend.app.models import Run
from backend.app.schemas import RunStatus
from backend.app.workers import local_worker as worker_module
from backend.app.workers.local_worker import LocalWorker


def wait_for_terminal_run(
    client: TestClient,
    run_id: str,
    timeout_seconds: float = 2.0,
) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        response = client.get(f"/runs/{run_id}")
        assert response.status_code == 200
        run = response.json()
        if run["status"] in {
            RunStatus.COMPLETED.value,
            RunStatus.FAILED.value,
        }:
            return run
        time.sleep(0.01)
    raise AssertionError("Run did not reach a terminal status")


def test_submit_and_complete_run(placeholder_client: TestClient) -> None:
    response = placeholder_client.post(
        "/runs",
        json={"prompt": "  Hello from the TTS system  ", "nfe": 8},
    )

    assert response.status_code == 202
    submitted_run = response.json()
    assert submitted_run["prompt"] == "Hello from the TTS system"
    assert submitted_run["nfe"] == 8
    assert submitted_run["status"] == RunStatus.QUEUED.value

    completed_run = wait_for_terminal_run(placeholder_client, submitted_run["id"])
    assert completed_run["status"] == RunStatus.COMPLETED.value
    assert completed_run["started_at"] is not None
    assert completed_run["completed_at"] is not None
    assert completed_run["error_message"] is None
    assert completed_run["result"] == {
        "message": "Placeholder processing completed",
        "processor": "local-placeholder",
        "received_prompt": "Hello from the TTS system",
        "nfe": 8,
    }


def test_list_runs(client: TestClient) -> None:
    submitted = client.post(
        "/runs",
        json={"prompt": "List this run", "nfe": 12},
    ).json()
    response = client.get("/runs?limit=10&offset=0")

    assert response.status_code == 200
    assert any(run["id"] == submitted["id"] for run in response.json())


@pytest.mark.parametrize("payload", [
    {"prompt": "   ", "nfe": 8},
    {"prompt": "Test", "nfe": 3},
    {"prompt": "Test", "nfe": 33},
    {"prompt": "Test", "nfe": 20.5},
    {"prompt": "Test", "nfe": "20"},
    {"prompt": "Test", "nfe": True},
    {"prompt": "x" * 5001, "nfe": 20},
    {"nfe": 20},
    {"prompt": "Test"},
])
def test_rejects_invalid_run_requests(client: TestClient, payload: Any) -> None:
    response = client.post("/runs", json=payload)
    assert response.status_code == 422
    assert client.get("/queue").json()["count"] == 0
    with SessionLocal() as database:
        assert database.scalar(select(func.count()).select_from(Run)) == 0


def test_unknown_run_returns_404(client: TestClient) -> None:
    response = client.get(f"/runs/{uuid4()}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Run not found"}


def test_worker_failure_is_persisted(
    placeholder_client: TestClient,
    monkeypatch: Any,
) -> None:
    def fail_processing(_run: Any) -> Dict[str, Any]:
        raise RuntimeError("Simulated processing failure")

    monkeypatch.setattr(
        worker_module,
        "build_placeholder_result",
        fail_processing,
    )
    submitted = placeholder_client.post(
        "/runs",
        json={"prompt": "This run should fail", "nfe": 4},
    ).json()

    failed_run = wait_for_terminal_run(placeholder_client, submitted["id"])
    assert failed_run["status"] == RunStatus.FAILED.value
    assert failed_run["error_message"] == "Simulated processing failure"


def test_local_frontend_origin_is_allowed(client: TestClient) -> None:
    response = client.options(
        "/runs",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == (
        "http://localhost:5173"
    )


@pytest.mark.parametrize("nfe", [4, 8, 20, 32])
def test_default_submission_is_persisted_and_stays_in_real_queue(
    client: TestClient, nfe: int,
) -> None:
    response = client.post("/runs", json={"prompt": "  Actual queued work  ", "nfe": nfe})
    assert response.status_code == 202
    submitted = response.json()
    assert submitted["prompt"] == "Actual queued work"
    assert submitted["status"] == "queued"

    with SessionLocal() as database:
        saved = database.get(Run, submitted["id"])
        assert saved is not None
        assert saved.prompt == "Actual queued work"
        assert saved.nfe == nfe
        assert saved.status == RunStatus.QUEUED
        assert saved.result is None
        assert saved.started_at is None
        assert saved.completed_at is None

    expected = {"mode": "queue-only", "count": 1, "run_ids": [submitted["id"]]}
    assert client.get("/queue").json() == expected
    assert client.get("/queue").json() == expected  # Inspection never consumes work.
    assert client.get(f"/runs/{submitted['id']}").json()["status"] == "queued"


def test_restart_restores_queued_rows_in_order_without_duplicates() -> None:
    first_worker = LocalWorker()
    with TestClient(create_app(first_worker)) as first:
        assert first.get("/runs").json() == []
        assert first.get("/queue").json()["count"] == 0
        submitted = [first.post("/runs", json={"prompt": text, "nfe": 20}).json()
                     for text in ["First queued run", "Second queued run"]]
        first_worker.start()  # An idempotent start must not enqueue duplicates.
        assert first.get("/queue").json()["run_ids"] == [run["id"] for run in submitted]

    restarted_worker = LocalWorker()
    restarted_app = create_app(restarted_worker)
    for _ in range(2):
        with TestClient(restarted_app) as restarted:
            assert restarted.get("/queue").json() == {
                "mode": "queue-only", "count": 2,
                "run_ids": [run["id"] for run in submitted],
            }
            assert all(run["status"] == "queued" for run in restarted.get("/runs").json())


def test_interrupted_processing_is_failed_on_restart_instead_of_replayed() -> None:
    with TestClient(create_app(LocalWorker())) as first:
        submitted = first.post("/runs", json={"prompt": "Interrupted work", "nfe": 4}).json()
        with SessionLocal() as database:
            repository.mark_run_running(database, submitted["id"])

    with TestClient(create_app(LocalWorker())) as restarted:
        run = restarted.get(f"/runs/{submitted['id']}").json()
        assert run["status"] == "failed"
        assert "interrupted" in run["error_message"]
        assert run["result"] is None
        assert restarted.get("/queue").json()["count"] == 0


def test_stopped_queue_rejects_submission_without_creating_a_row(
    client: TestClient, worker: LocalWorker,
) -> None:
    worker.stop()
    response = client.post("/runs", json={"prompt": "Do not strand this request", "nfe": 20})
    assert response.status_code == 503
    assert client.get("/runs").json() == []
    assert client.get("/queue").json()["count"] == 0
