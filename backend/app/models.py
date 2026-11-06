from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum as SQLAlchemyEnum,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database import Base
from backend.app.schemas import RunStatus


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Run(Base):
    __tablename__ = "runs"

    __table_args__ = (
        CheckConstraint(
            "nfe >= 4 AND nfe <= 32",
            name="check_nfe_range",
        ),
        CheckConstraint(
            "length(trim(prompt)) > 0",
            name="check_prompt_not_empty",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )

    prompt: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    nfe: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    status: Mapped[RunStatus] = mapped_column(
        SQLAlchemyEnum(
            RunStatus,
            values_callable=lambda statuses: [
                status.value for status in statuses
            ],
            native_enum=False,
            create_constraint=True,
            name="run_status",
        ),
        nullable=False,
        default=RunStatus.QUEUED,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    result: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
    )

    audio_path: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True,
    )

    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )