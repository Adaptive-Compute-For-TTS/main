import time
from typing import Any, Dict
from uuid import uuid4

from fastapi.testclient import TestClient

from backend.app.schemas import RunStatus
from backend.app.workers import local_worker as worker_module


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


def test_submit_and_complete_run(client: TestClient) -> None:
    response = client.post(
        "/runs",
        json={"prompt": "  Hello from the TTS system  ", "nfe": 8},
    )

    assert response.status_code == 202
    submitted_run = response.json()
    assert submitted_run["prompt"] == "Hello from the TTS system"
    assert submitted_run["nfe"] == 8
    assert submitted_run["status"] == RunStatus.QUEUED.value

    completed_run = wait_for_terminal_run(client, submitted_run["id"])
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
    wait_for_terminal_run(client, submitted["id"])

    response = client.get("/runs?limit=10&offset=0")

    assert response.status_code == 200
    assert any(run["id"] == submitted["id"] for run in response.json())


def test_rejects_invalid_run_requests(client: TestClient) -> None:
    empty_prompt = client.post("/runs", json={"prompt": "   ", "nfe": 8})
    low_nfe = client.post("/runs", json={"prompt": "Test", "nfe": 3})
    high_nfe = client.post("/runs", json={"prompt": "Test", "nfe": 33})

    assert empty_prompt.status_code == 422
    assert low_nfe.status_code == 422
    assert high_nfe.status_code == 422


def test_unknown_run_returns_404(client: TestClient) -> None:
    response = client.get(f"/runs/{uuid4()}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Run not found"}


def test_worker_failure_is_persisted(
    client: TestClient,
    monkeypatch: Any,
) -> None:
    def fail_processing(_run: Any) -> Dict[str, Any]:
        raise RuntimeError("Simulated processing failure")

    monkeypatch.setattr(
        worker_module,
        "build_placeholder_result",
        fail_processing,
    )
    submitted = client.post(
        "/runs",
        json={"prompt": "This run should fail", "nfe": 4},
    ).json()

    failed_run = wait_for_terminal_run(client, submitted["id"])
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
