from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


Prompt = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=5000,
    ),
]


class RunCreate(BaseModel):
    prompt: Prompt
    nfe: int = Field(ge=4, le=32)


class RunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    prompt: str
    nfe: int
    status: RunStatus

    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    result: Optional[Dict[str, Any]] = None
    audio_path: Optional[str] = None
    error_message: Optional[str] = None