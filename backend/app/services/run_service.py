from typing import List, Optional

from sqlalchemy.orm import Session

from backend.app import repository
from backend.app.models import Run
from backend.app.schemas import RunCreate
from backend.app.workers.local_worker import local_worker


def submit_run(database: Session, run_data: RunCreate) -> Run:
    run = repository.create_run(database, run_data)
    local_worker.submit(run.id)
    return run


def get_run(database: Session, run_id: str) -> Optional[Run]:
    return repository.get_run(database, run_id)


def list_runs(
    database: Session,
    *,
    offset: int = 0,
    limit: int = 20,
) -> List[Run]:
    return repository.list_runs(database, offset=offset, limit=limit)
