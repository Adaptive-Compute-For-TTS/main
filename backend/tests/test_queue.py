from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Any

from fastapi.testclient import TestClient

from backend.app import repository
from backend.app.workers.local_worker import LocalWorker


def test_shutdown_waits_for_a_committed_submission_to_enter_the_queue(
    client: TestClient, worker: LocalWorker, monkeypatch: Any,
) -> None:
    committed, release, stop_attempted, stopped = (Event() for _ in range(4))
    create_run = repository.create_run

    def commit_then_pause(*args: Any, **kwargs: Any) -> Any:
        run = create_run(*args, **kwargs)
        committed.set()
        assert release.wait(timeout=2)
        return run

    def stop_queue() -> None:
        stop_attempted.set()
        worker.stop()
        stopped.set()

    monkeypatch.setattr(repository, "create_run", commit_then_pause)
    with ThreadPoolExecutor(max_workers=2) as executor:
        submission = executor.submit(
            client.post, "/runs", json={"prompt": "Keep this committed request", "nfe": 20},
        )
        try:
            assert committed.wait(timeout=2)
            shutdown = executor.submit(stop_queue)
            assert stop_attempted.wait(timeout=2)
            assert not stopped.wait(timeout=0.05)
        finally:
            release.set()

        response = submission.result(timeout=2)
        shutdown.result(timeout=2)

    assert response.status_code == 202
    run_id = response.json()["id"]
    assert client.get(f"/runs/{run_id}").json()["status"] == "queued"
    worker.start()
    assert client.get("/queue").json()["run_ids"] == [run_id]
