from typing import List, Optional

from sqlalchemy.orm import Session

from backend.app import repository
from backend.app.models import Run
from backend.app.schemas import RunCreate
from backend.app.workers.local_worker import LocalWorker


def submit_run(database: Session, run_data: RunCreate, worker: LocalWorker) -> Run:
    return worker.create_and_submit(database, run_data)


def get_run(database: Session, run_id: str) -> Optional[Run]:
    return repository.get_run(database, run_id)


def list_runs(
    database: Session,
    *,
    offset: int = 0,
    limit: int = 20,
) -> List[Run]:
    return repository.list_runs(database, offset=offset, limit=limit)
