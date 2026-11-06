from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.schemas import RunCreate, RunResponse
from backend.app.services import run_service


router = APIRouter(prefix="/runs", tags=["Runs"])


@router.post(
    "",
    response_model=RunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def submit_run(
    run_data: RunCreate,
    database: Session = Depends(get_db),
) -> RunResponse:
    run = run_service.submit_run(database, run_data)
    return RunResponse.model_validate(run)


@router.get("", response_model=List[RunResponse])
def get_runs(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    database: Session = Depends(get_db),
) -> List[RunResponse]:
    runs = run_service.list_runs(database, offset=offset, limit=limit)
    return [RunResponse.model_validate(run) for run in runs]


@router.get("/{run_id}", response_model=RunResponse)
def get_run(
    run_id: UUID,
    database: Session = Depends(get_db),
) -> RunResponse:
    run = run_service.get_run(database, str(run_id))
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run not found",
        )
    return RunResponse.model_validate(run)
