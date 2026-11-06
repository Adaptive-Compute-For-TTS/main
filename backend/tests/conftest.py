import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete


TEST_DATABASE_PATH = (
    Path(tempfile.gettempdir())
    / f"adaptive_compute_tts_test_{os.getpid()}.db"
)
os.environ["ADAPTIVE_TTS_DATABASE_URL"] = (
    f"sqlite:///{TEST_DATABASE_PATH}"
)
os.environ["ADAPTIVE_TTS_PROCESSING_DELAY_SECONDS"] = "0.01"

# Application imports must follow the test environment configuration above.
from backend.app.database import SessionLocal, engine, init_db  # noqa: E402
from backend.app.main import app  # noqa: E402
from backend.app.models import Run  # noqa: E402
from backend.app.workers.local_worker import local_worker  # noqa: E402


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def reset_runs() -> Generator[None, None, None]:
    init_db()
    local_worker.wait_until_idle()
    with SessionLocal() as database:
        database.execute(delete(Run))
        database.commit()

    yield

    local_worker.wait_until_idle()


def pytest_sessionfinish() -> None:
    engine.dispose()
    TEST_DATABASE_PATH.unlink(missing_ok=True)
