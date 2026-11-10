"""The backend contract: *where* jobs run.

A backend is a context manager (provision on enter, release on exit) that ships a `Job` to an
environment running `ttsbench.worker.execute_job` and returns its `JobResult`. Backends must
not time anything themselves; all latency numbers come from the worker.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any

from ttsbench.schema import ComponentSpec, Job, JobResult


class Backend(ABC):
    def __init__(self, model: ComponentSpec) -> None:
        self.model = model

    def __enter__(self) -> Backend:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    @abstractmethod
    def execute(self, job: Job) -> JobResult: ...

    def describe(self) -> dict[str, Any]:
        """Backend facts stored in the run manifest (GPU type, region, image, ...)."""
        return {}
