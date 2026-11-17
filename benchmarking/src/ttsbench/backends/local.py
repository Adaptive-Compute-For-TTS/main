"""Run jobs in the current process (a local GPU box, a dev laptop, or CI)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ttsbench.backends.base import Backend
from ttsbench.schema import ComponentSpec, Job, JobResult
from ttsbench.worker import execute_job


class LocalBackend(Backend):
    def __init__(self, model: ComponentSpec, assets_root: str = "benchmarks") -> None:
        super().__init__(model)
        self.assets_root = Path(assets_root)

    def execute(self, job: Job) -> JobResult:
        return execute_job(job, assets_root=self.assets_root)

    def describe(self) -> dict[str, Any]:
        return {"assets_root": str(self.assets_root)}
