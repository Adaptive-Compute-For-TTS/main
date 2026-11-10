from __future__ import annotations

import json
from pathlib import Path

import pytest

from ttsbench.schema import ComponentSpec, ExperimentConfig, MeasurementConfig, Prompt

PROMPTS = [
    {"id": "s1", "text": "Short one.", "length": "short", "tags": ["plain"]},
    {"id": "m1", "text": "A medium length sentence for testing purposes.", "length": "medium", "tags": ["plain"]},
    {"id": "l1", "text": "A considerably longer sentence " * 4, "length": "long", "tags": ["numbers"]},
]


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    prompts_dir = tmp_path / "benchmarks" / "prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "v0.jsonl").write_text("\n".join(json.dumps(p) for p in PROMPTS) + "\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def prompts() -> list[Prompt]:
    return [Prompt.model_validate(p) for p in PROMPTS]


@pytest.fixture
def config(workspace: Path) -> ExperimentConfig:
    return ExperimentConfig(
        name="test sine",
        model=ComponentSpec(name="sine"),
        sweep={"nfe": (2, 4)},
        measurement=MeasurementConfig(warmup=1, repeats=3),
        output_dir=str(workspace / "runs"),
    )
