from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, computed_field, field_validator


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
    nfe: int = Field(ge=4, le=32, strict=True)


class QueueStatus(BaseModel):
    mode: str
    count: int
    run_ids: list[str]


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


    @field_validator("created_at", "updated_at", "started_at", "completed_at", mode="before")
    @classmethod
    def utc_timestamps(cls, value):
        # SQLite drops timezone metadata from DateTime columns. Our writer uses
        # UTC, so serialize the UTC offset instead of making browsers guess.
        if isinstance(value, datetime) and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @computed_field
    @property
    def audio_url(self) -> Optional[str]:
        if self.status == RunStatus.COMPLETED and self.audio_path:
            return f"/runs/{self.id}/audio"
        return None
