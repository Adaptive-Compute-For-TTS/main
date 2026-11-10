"""Run jobs on a Modal GPU container.

The app is built per run from the model adapter's `Requirements`, so adding a model never
touches this file. All jobs of a run go to a single container (`max_containers=1`) to keep
hardware identical across NFE values, and the container stays warm between jobs.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any

from ttsbench.backends.base import Backend
from ttsbench.models import MODELS
from ttsbench.schema import ComponentSpec, Job, JobResult
from ttsbench.worker import ASSETS_ROOT_ENV, execute_job_payload

_REMOTE_ASSETS = "/assets"
_HF_CACHE = "/cache/hf"
_BUILD_DIR = "/opt/ttsbench-build"
_BASE_PACKAGES = ("numpy>=1.26", "pydantic>=2.10", "pyyaml>=6.0", "soundfile>=0.12")


class ModalBackend(Backend):
    def __init__(
        self,
        model: ComponentSpec,
        gpu: str | None = "H100",
        timeout_s: int = 3600,
        region: str | None = None,
        app_name: str = "ttsbench",
        cache_volume: str = "ttsbench-hf-cache",
        assets_root: str = "benchmarks",
        show_logs: bool = True,
    ) -> None:
        super().__init__(model)
        self.gpu = gpu
        self.timeout_s = timeout_s
        self.region = region
        self.app_name = app_name
        self.cache_volume = cache_volume
        self.assets_root = Path(assets_root)
        self.show_logs = show_logs
        self._stack: ExitStack | None = None
        self._fn: Any = None

    def _build(self) -> tuple[Any, Any]:
        import modal

        req = MODELS.resolve(self.model.name).requirements
        image = modal.Image.debian_slim(python_version=req.python)
        if req.apt:
            image = image.apt_install(*req.apt)
        # Model pins first so ttsbench's loose base requirements never override them.
        if req.pip:
            image = image.pip_install(*req.pip)
        for local, name in req.sources:
            remote = f"{_BUILD_DIR}/{name}"
            if local.is_dir():
                image = image.add_local_dir(local, remote, copy=True, ignore=["**/__pycache__"])
            else:
                image = image.add_local_file(local, remote, copy=True)
        if req.setup:
            image = image.run_commands(*req.setup_commands(build=_BUILD_DIR, pip="python -m pip install"))
        image = image.pip_install(*_BASE_PACKAGES)
        image = image.env({ASSETS_ROOT_ENV: _REMOTE_ASSETS, "HF_HOME": _HF_CACHE})
        if self.assets_root.is_dir():
            image = image.add_local_dir(self.assets_root, _REMOTE_ASSETS)
        image = image.add_local_python_source("ttsbench")

        app = modal.App(self.app_name)
        fn = app.function(
            image=image,
            gpu=self.gpu,
            timeout=self.timeout_s,
            region=self.region,
            volumes={"/cache": modal.Volume.from_name(self.cache_volume, create_if_missing=True)},
            max_containers=1,
            scaledown_window=300,
            serialized=True,
        )(execute_job_payload)
        return app, fn

    def __enter__(self) -> ModalBackend:
        import modal

        app, fn = self._build()
        with ExitStack() as stack:
            if self.show_logs:
                stack.enter_context(modal.enable_output())
            stack.enter_context(app.run())
            self._stack = stack.pop_all()
        self._fn = fn
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self._stack is not None:
            self._stack.__exit__(*exc_info)
            self._stack = None
        self._fn = None

    def execute(self, job: Job) -> JobResult:
        if self._fn is None:
            raise RuntimeError("ModalBackend must be used as a context manager")
        return JobResult.model_validate(self._fn.remote(job.model_dump()))

    def describe(self) -> dict[str, Any]:
        return {"gpu": self.gpu, "region": self.region, "app_name": self.app_name}
