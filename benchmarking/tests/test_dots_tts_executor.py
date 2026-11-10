"""CPU checks for the dots_tts_executor model: dots.tts and adaptive_nfe are stubbed.

The real path (patched dots.tts + executor on a GPU) is exercised by running
configs/executor-modal.yaml; these tests pin the wiring around it.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from ttsbench.measure.timing import Device, StageTimer
from ttsbench.models import MODELS
from ttsbench.models.base import SynthesisRequest
from ttsbench.models.dots_tts_executor import DOTS_REVISION, HOOK_PATCH, DotsTTSExecutor


class FakeRuntime:
    def __init__(self) -> None:
        self.model = object()
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"audio": [0.0] * 240, "sample_rate": 24_000}


@dataclass
class FakeResult:
    output: dict[str, Any]
    nfe_per_patch: list[int]

    @property
    def num_patches(self) -> int:
        return len(self.nfe_per_patch)


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stand-ins for dots_tts.runtime, loguru and adaptive_nfe that record how they were called."""
    seen: dict[str, Any] = {"fixed_nfe": [], "executors": []}
    runtime = FakeRuntime()

    class DotsTtsRuntime:
        @staticmethod
        def from_pretrained(checkpoint: str, **kwargs: Any) -> FakeRuntime:
            seen["from_pretrained"] = (checkpoint, kwargs)
            return runtime

    class Executor:
        def __init__(self, model: Any) -> None:
            seen["executors"].append(model)

    def generate_fixed_nfe(rt: Any, nfe: int, *, use_executor: bool, executor: Any, **kwargs: Any) -> FakeResult:
        seen["fixed_nfe"].append({"nfe": nfe, "use_executor": use_executor, "executor": executor, **kwargs})
        return FakeResult(rt.generate(num_steps=nfe, **kwargs), [nfe] * 3)

    modules = {
        "dots_tts": types.ModuleType("dots_tts"),
        "dots_tts.runtime": types.SimpleNamespace(DotsTtsRuntime=DotsTtsRuntime),
        "loguru": types.SimpleNamespace(logger=types.SimpleNamespace(remove=lambda: None, add=lambda *a, **k: None)),
        "adaptive_nfe": types.SimpleNamespace(generate_fixed_nfe=generate_fixed_nfe),
        "adaptive_nfe.executor": types.SimpleNamespace(Executor=Executor),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    seen["runtime"] = runtime
    return seen


def _synthesize(model: DotsTTSExecutor, nfe: int = 10):
    return model.synthesize(SynthesisRequest(text="Hello there.", nfe=nfe, seed=0), StageTimer(Device("cpu")))


def test_registered_and_rejects_unknown_mode() -> None:
    assert MODELS.resolve("dots_tts_executor") is DotsTTSExecutor
    with pytest.raises(ValueError, match="mode must be one of"):
        DotsTTSExecutor(mode="adaptive")


def test_requirements_ship_the_patch_and_executor_sources() -> None:
    req = DotsTTSExecutor.requirements
    names = [name for _, name in req.sources]
    assert all(local.exists() for local, _ in req.sources), "gpu-executor files moved; update the sources"
    assert any(HOOK_PATCH.startswith(name) for name in names)
    setup = req.setup_commands(build="/b", pip="pip install")
    assert f'git -C "/b/dots.tts" checkout -q {DOTS_REVISION}' in setup
    assert f'git -C "/b/dots.tts" apply "/b/{HOOK_PATCH}"' in setup
    assert setup[-1] == 'pip install "/b/dots.tts" "/b/gpu-executor"'
    assert not any(p.startswith("dots.tts") for p in req.pip), "dots.tts must come from the patched clone"


def test_sources_digest_changes_with_content(tmp_path) -> None:
    from ttsbench.models.base import Requirements

    (tmp_path / "a.txt").write_text("one")
    req = Requirements(sources=((tmp_path, "src"),))
    before = req.sources_digest()
    (tmp_path / "a.txt").write_text("two")
    assert req.sources_digest() != before


def test_executor_mode_runs_generate_fixed_nfe_with_one_reused_executor(stubs: dict[str, Any]) -> None:
    model = DotsTTSExecutor(mode="executor", optimize=True)
    model.load()
    assert stubs["from_pretrained"] == ("dots-studio/dots.tts-soar", {"precision": "bfloat16", "optimize": True})

    first, second = _synthesize(model, nfe=10), _synthesize(model, nfe=4)

    calls = stubs["fixed_nfe"]
    assert [c["nfe"] for c in calls] == [10, 4] and all(c["use_executor"] for c in calls)
    assert len(stubs["executors"]) == 1 and calls[0]["executor"] is calls[1]["executor"] is not None
    assert first.extras == {"num_patches": 3, "nfe_per_patch": [10, 10, 10]}
    assert second.sample_rate == 24_000 and calls[0]["guidance_scale"] == 1.2


def test_scheduler_mode_keeps_the_dots_loop(stubs: dict[str, Any]) -> None:
    model = DotsTTSExecutor(mode="scheduler")
    model.load()
    _synthesize(model)
    assert stubs["executors"] == []
    assert stubs["fixed_nfe"][0]["use_executor"] is False and stubs["fixed_nfe"][0]["executor"] is None


def test_stock_mode_calls_generate_directly(stubs: dict[str, Any]) -> None:
    model = DotsTTSExecutor(mode="stock")
    model.load()
    output = _synthesize(model, nfe=16)
    assert stubs["fixed_nfe"] == [] and output.extras == {}
    assert stubs["runtime"].calls == [{"num_steps": 16, "text": "Hello there.", "guidance_scale": 1.2}]
    assert model.info()["mode"] == "stock" and model.info()["dots_tts_revision"] == DOTS_REVISION


def test_imports_from_a_shallow_install(tmp_path) -> None:
    """In a Modal container ttsbench lives at /root/ttsbench: importing must not need the repo."""
    import shutil
    import subprocess

    import ttsbench

    shutil.copytree(Path(ttsbench.__file__).parent, tmp_path / "ttsbench", ignore=shutil.ignore_patterns("__pycache__"))
    code = "from ttsbench.models import MODELS; print(MODELS.create('dots_tts_executor').mode)"
    proc = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(tmp_path)},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "executor"
