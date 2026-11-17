from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ttsbench.config import apply_override, config_hash, load_config
from ttsbench.prompts import load_prompt_file, prompts_hash, select_prompts
from ttsbench.schema import ExperimentConfig, Prompt, PromptSelection, SweepConfig


def test_load_config_with_overrides(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("name: x\nmodel: {name: sine}\nbackend: {name: local}\nsweep: {nfe: [4]}\n")
    config = load_config(path, ["sweep.nfe=[2,8]", "backend.params.gpu=A100", "measurement.repeats=9"])
    assert config.sweep.nfe == (2, 8)
    assert config.backend.params == {"gpu": "A100"}
    assert config.measurement.repeats == 9


def test_override_does_not_mutate_input() -> None:
    raw = {"a": {"b": 1}}
    assert apply_override(raw, "a.b=2") == {"a": {"b": 2}}
    assert raw == {"a": {"b": 1}}


@pytest.mark.parametrize("bad", ["no_equals", "=3"])
def test_bad_override(bad: str) -> None:
    with pytest.raises(ValueError):
        apply_override({}, bad)


def test_config_hash_ignores_output_dir_but_not_settings(config: ExperimentConfig) -> None:
    moved = config.model_copy(update={"output_dir": "elsewhere"})
    changed = config.model_copy(update={"sweep": SweepConfig(nfe=(2, 4, 8))})
    assert config_hash(config) == config_hash(moved)
    assert config_hash(config) != config_hash(changed)


@pytest.mark.parametrize("nfe", [(), (0,), (4, 4)])
def test_invalid_sweeps(nfe: tuple[int, ...]) -> None:
    with pytest.raises(ValidationError):
        SweepConfig(nfe=nfe)


def test_unknown_config_keys_rejected() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"name": "x", "model": {"name": "sine"}, "sweep": {"nfe": [1]}, "nfee": 1})


def test_select_prompts_filters(workspace: Path) -> None:
    assert [p.id for p in select_prompts(PromptSelection(lengths=("short", "long")))] == ["s1", "l1"]
    assert [p.id for p in select_prompts(PromptSelection(tags=("numbers",)))] == ["l1"]
    assert [p.id for p in select_prompts(PromptSelection(limit=1))] == ["s1"]
    with pytest.raises(ValueError, match="unknown prompt ids"):
        select_prompts(PromptSelection(ids=("nope",)))


def test_prompt_file_validation(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    row = '{"id": "a", "text": "t", "length": "short"}'
    path.write_text(f"# comment\n{row}\n{row}\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_prompt_file(path)
    path.write_text('{"id": "../evil", "text": "t", "length": "short"}\n')
    with pytest.raises(ValueError, match="filesystem-safe"):
        load_prompt_file(path)


def test_prompts_hash_changes_with_text(prompts: list[Prompt]) -> None:
    edited = [prompts[0].model_copy(update={"text": "different"}), *prompts[1:]]
    assert prompts_hash(prompts) != prompts_hash(edited)


def test_repo_prompt_set_is_valid() -> None:
    prompts = load_prompt_file(Path(__file__).parents[1] / "benchmarks" / "prompts" / "v0.jsonl")
    assert {p.length for p in prompts} == {"short", "medium", "long"}
