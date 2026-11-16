"""Real HTTP checks of Next.js -> FastAPI -> SQLite + queue.Queue.

Requires `npm run build` in frontend/. All writes use pytest's temporary
SQLite file. The development database is never opened by these tests.
"""

import json
import os
from pathlib import Path
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import pytest


ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def fetch(base, path, body=None):
    request = Request(
        base + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET",
    )
    try:
        response = urlopen(request, timeout=15)
    except HTTPError as error:
        response = error
    with response:
        content = response.read()
        return response.status, json.loads(content) if content else None


def stop(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


class LocalSystem:
    def __init__(self, directory):
        self.directory = directory
        self.database = directory / "integration.db"
        self.backend_port = free_port()
        self.frontend_port = free_port()
        while self.frontend_port == self.backend_port:
            self.frontend_port = free_port()
        self.backend_url = f"http://127.0.0.1:{self.backend_port}"
        self.frontend_url = f"http://127.0.0.1:{self.frontend_port}"
        self.backend = None
        self.frontend = None
        self.logs = []

    def launch(self, command, cwd, environment, name):
        log = (self.directory / f"{name}.log").open("a")
        self.logs.append(log)
        return subprocess.Popen(
            command, cwd=cwd, env=environment, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )

    def wait_ready(self, process, base, path):
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            try:
                if fetch(base, path)[0] == 200:
                    return
            except (URLError, TimeoutError, json.JSONDecodeError):
                pass
            time.sleep(0.1)
        logs = "\n".join(path.read_text() for path in self.directory.glob("*.log"))
        raise AssertionError(f"Local test server failed to start:\n{logs}")

    def start_backend(self):
        environment = os.environ.copy()
        environment.update({
            "ADAPTIVE_TTS_DATABASE_URL": f"sqlite:///{self.database}",
            "ADAPTIVE_TTS_WORKER_MODE": "queue-only",
        })
        self.backend = self.launch(
            [sys.executable, "-m", "uvicorn", "backend.app.main:app",
             "--host", "127.0.0.1", "--port", str(self.backend_port)],
            ROOT, environment, "backend",
        )
        self.wait_ready(self.backend, self.backend_url, "/health")

    def start(self):
        assert (ROOT / "frontend/.next/BUILD_ID").exists(), "Build frontend first: cd frontend && npm run build"
        self.start_backend()
        environment = os.environ.copy()
        environment.update({"FASTAPI_BASE_URL": self.backend_url, "NEXT_TELEMETRY_DISABLED": "1"})
        self.frontend = self.launch(
            [shutil.which("node"), "node_modules/next/dist/bin/next", "start",
             "--hostname", "127.0.0.1", "--port", str(self.frontend_port)],
            ROOT / "frontend", environment, "frontend",
        )
        self.wait_ready(self.frontend, self.frontend_url, "/api/runs")
        assert fetch(self.backend_url, "/runs") == (200, [])
        assert fetch(self.backend_url, "/queue") == (200, {"mode": "queue-only", "count": 0, "run_ids": []})

    def rows(self):
        with sqlite3.connect(f"file:{self.database}?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute("SELECT * FROM runs ORDER BY created_at, id")]

    def close(self):
        stop(self.frontend)
        stop(self.backend)
        for log in self.logs:
            log.close()


@pytest.fixture(scope="module")
def system(tmp_path_factory):
    instance = LocalSystem(tmp_path_factory.mktemp("frontend-fastapi-sqlite"))
    try:
        instance.start()
        yield instance
    finally:
        instance.close()


@pytest.mark.parametrize("nfe", [4, 20, 32])
def test_frontend_request_is_persisted_and_queued(system, nfe):
    before = len(system.rows())
    prompt = f"  HTTP integration request at NFE {nfe}.  "
    status, run = fetch(system.frontend_url, "/api/runs", {"prompt": prompt, "nfe": nfe})
    assert status == 202
    assert run["status"] == "queued"
    assert run["prompt"] == prompt.strip()
    assert run["nfe"] == nfe
    assert run["result"] is None
    assert run["started_at"] is None
    assert run["completed_at"] is None
    rows = system.rows()
    assert len(rows) == before + 1
    persisted = next(row for row in rows if row["id"] == run["id"])
    assert (persisted["prompt"], persisted["nfe"], persisted["status"]) == (prompt.strip(), nfe, "queued")
    queue_status, queue = fetch(system.backend_url, "/queue")
    assert queue_status == 200
    assert queue["count"] == len(rows)
    assert queue["run_ids"][-1] == run["id"]
    assert fetch(system.frontend_url, f"/api/runs?id={run['id']}") == (200, run)


def test_invalid_requests_create_neither_rows_nor_jobs(system):
    before_rows = system.rows()
    before_queue = fetch(system.backend_url, "/queue")
    for body, expected_status in [
        ({"prompt": " \n ", "nfe": 20}, 400),
        ({"prompt": "Invalid selection", "nfe": 8}, 400),
        ({"prompt": "x" * 5001, "nfe": 4}, 422),
    ]:
        status, error = fetch(system.frontend_url, "/api/runs", body)
        assert status == expected_status
        assert error
    assert fetch(system.frontend_url, f"/api/runs?id={uuid4()}")[0] == 404
    assert fetch(system.frontend_url, "/api/runs?id=not-a-uuid")[0] == 400
    assert system.rows() == before_rows
    assert fetch(system.backend_url, "/queue") == before_queue


def test_backend_restart_restores_queued_jobs_from_sqlite(system):
    before_rows = system.rows()
    before_queue = fetch(system.backend_url, "/queue")
    stop(system.backend)
    system.start_backend()
    assert system.rows() == before_rows
    assert fetch(system.backend_url, "/queue") == before_queue
    status, run = fetch(system.frontend_url, "/api/runs", {"prompt": "Request after backend restart.", "nfe": 20})
    assert status == 202
    assert run["status"] == "queued"
    assert fetch(system.backend_url, "/queue")[1]["run_ids"][-1] == run["id"]
    assert len(system.rows()) == len(before_rows) + 1
