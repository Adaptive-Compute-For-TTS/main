from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import Run
from backend.app.schemas import RunCreate, RunStatus


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def create_run(database: Session, run_data: RunCreate) -> Run:
    run = Run(
        prompt=run_data.prompt,
        nfe=run_data.nfe,
        status=RunStatus.QUEUED,
    )
    database.add(run)
    database.commit()
    database.refresh(run)
    return run


def get_run(database: Session, run_id: str) -> Optional[Run]:
    return database.get(Run, run_id)


def list_runs(
    database: Session,
    *,
    offset: int = 0,
    limit: int = 20,
) -> List[Run]:
    statement = (
        select(Run)
        .order_by(Run.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(database.scalars(statement).all())


def mark_run_running(database: Session, run_id: str) -> Optional[Run]:
    run = get_run(database, run_id)
    if run is None or run.status != RunStatus.QUEUED:
        return None

    run.status = RunStatus.RUNNING
    run.started_at = utc_now()
    run.error_message = None
    database.commit()
    database.refresh(run)
    return run


def mark_run_completed(
    database: Session,
    run_id: str,
    result: Dict[str, Any],
    audio_path: Optional[str] = None,
) -> Optional[Run]:
    run = get_run(database, run_id)
    if run is None or run.status != RunStatus.RUNNING:
        return None

    run.status = RunStatus.COMPLETED
    run.result = result
    run.audio_path = audio_path
    run.completed_at = utc_now()
    database.commit()
    database.refresh(run)
    return run


def mark_run_failed(
    database: Session,
    run_id: str,
    error_message: str,
) -> Optional[Run]:
    run = get_run(database, run_id)
    if run is None or run.status not in {
        RunStatus.QUEUED,
        RunStatus.RUNNING,
    }:
        return None

    run.status = RunStatus.FAILED
    run.error_message = error_message
    run.completed_at = utc_now()
    database.commit()
    database.refresh(run)
    return run
