from pathlib import Path
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session
from fastapi.responses import FileResponse
from backend.app.config import AUDIO_DIR
from backend.app.schemas import RunStatus

from backend.app.database import get_db
from backend.app.schemas import RunCreate, RunResponse
from backend.app.services import run_service
from backend.app.workers.local_worker import QueueUnavailableError


router = APIRouter(prefix="/runs", tags=["Runs"])


@router.post(
    "",
    response_model=RunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def submit_run(
    run_data: RunCreate,
    request: Request,
    database: Session = Depends(get_db),
) -> RunResponse:
    try:
        run = run_service.submit_run(database, run_data, request.app.state.run_queue)
    except QueueUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
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


@router.get("/{run_id}/audio", response_class=FileResponse)
def get_run_audio(run_id: UUID, database: Session = Depends(get_db)):
    run = run_service.get_run(database, str(run_id))
    if run is None or run.status != RunStatus.COMPLETED or not run.audio_path:
        raise HTTPException(status_code=404, detail="Audio is not available for this run")
    path = Path(run.audio_path).resolve()
    if not path.is_relative_to(AUDIO_DIR) or path.suffix.lower() != ".wav" or not path.is_file():
        raise HTTPException(status_code=404, detail="Audio file is unavailable")
    return FileResponse(path, media_type="audio/wav", filename=f"{run_id}.wav", content_disposition_type="inline")
