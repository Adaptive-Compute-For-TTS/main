from __future__ import annotations

import pytest

from ttsbench.schema import ComponentSpec, Job, MeasurementConfig

pytest.importorskip("modal")

from ttsbench.backends import BACKENDS  # noqa: E402
from ttsbench.backends.modal_backend import ModalBackend  # noqa: E402


def test_modal_backend_builds_app_offline() -> None:
    backend = BACKENDS.create("modal", model=ComponentSpec(name="dots_tts"), gpu="L40S")
    app, fn = backend._build()
    assert app is not None and fn is not None
    assert backend.describe()["gpu"] == "L40S"


def test_modal_backend_requires_context_manager() -> None:
    backend = ModalBackend(model=ComponentSpec(name="sine"))
    job = Job(model=backend.model, nfe=1, prompts=(), measurement=MeasurementConfig())
    with pytest.raises(RuntimeError, match="context manager"):
        backend.execute(job)
