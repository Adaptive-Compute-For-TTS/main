import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete


TEST_DIRECTORY = tempfile.TemporaryDirectory(prefix="adaptive-compute-tests-")
TEST_DATABASE_PATH = Path(TEST_DIRECTORY.name) / "runs.db"
os.environ["ADAPTIVE_TTS_DATABASE_URL"] = f"sqlite:///{TEST_DATABASE_PATH}"
os.environ["ADAPTIVE_TTS_PROCESSING_DELAY_SECONDS"] = "0.01"
os.environ.pop("ADAPTIVE_TTS_WORKER_MODE", None)

# Application imports must follow the isolated database configuration above.
from backend.app.database import SessionLocal, engine, init_db  # noqa: E402
from backend.app.main import create_app  # noqa: E402
from backend.app.models import Run  # noqa: E402
from backend.app.workers.local_worker import LocalWorker  # noqa: E402


@pytest.fixture
def worker() -> LocalWorker:
    return LocalWorker()


@pytest.fixture
def client(worker: LocalWorker) -> Generator[TestClient, None, None]:
    with TestClient(create_app(worker)) as test_client:
        yield test_client


@pytest.fixture
def placeholder_client() -> Generator[TestClient, None, None]:
    # Placeholder processing is opt-in only, even in the test suite.
    with TestClient(create_app(LocalWorker(mode="placeholder"))) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def reset_runs() -> Generator[None, None, None]:
    init_db()
    with SessionLocal() as database:
        database.execute(delete(Run))
        database.commit()
    yield


def pytest_sessionfinish() -> None:
    engine.dispose()
    TEST_DIRECTORY.cleanup()
