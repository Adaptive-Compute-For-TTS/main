from __future__ import annotations

from pathlib import Path

import pytest

from ttsbench.cli import main
from ttsbench.compare import compare_runs
from ttsbench.measure.timing import StageTimer
from ttsbench.models import MODELS, SynthesisOutput, SynthesisRequest, TTSModel
from ttsbench.runner import run_experiment
from ttsbench.schema import ComponentSpec, ExperimentConfig
from ttsbench.storage import list_runs


class ExplodingModel(TTSModel):
    device = "cpu"

    def load(self) -> None:
        pass

    def synthesize(self, request: SynthesisRequest, timer: StageTimer) -> SynthesisOutput:
        if request.nfe > 2:
            raise RuntimeError("kernel crashed")
        return SynthesisOutput(audio=[0.0] * 100, sample_rate=100)


MODELS.register("exploding", ExplodingModel)


def test_run_writes_complete_run_directory(config: ExperimentConfig) -> None:
    run = run_experiment(config)
    manifest = run.manifest

    assert manifest["status"] == "completed"
    assert manifest["measurement_protocol_version"] == 1
    assert manifest["config"]["sweep"]["nfe"] == [2, 4]
    assert manifest["config_hash"] in run.path.name or manifest["config_hash"][:8] in run.path.name
    assert {"git", "client", "backend", "prompts_hash"} <= manifest.keys()
    assert len(run.samples()) == 2 * 3 * 3
    assert [w["nfe"] for w in run.workers()] == [2, 4]
    assert run.workers()[0]["environment"]["python"]
    assert len(list((run.path / "audio").rglob("*.wav"))) == 2 * 3
    assert (run.path / "summary.csv").read_text().startswith("nfe,length,n,")


def test_failed_run_keeps_completed_nfe_values(config: ExperimentConfig) -> None:
    broken = config.model_copy(update={"model": ComponentSpec(name="exploding")})
    with pytest.raises(RuntimeError, match="kernel crashed"):
        run_experiment(broken)
    (run,) = list_runs(Path(config.output_dir))
    assert run.manifest["status"] == "failed"
    assert "kernel crashed" in run.manifest["error"]
    assert {s.nfe for s in run.samples()} == {2}


def test_unknown_components_fail_before_creating_a_run(config: ExperimentConfig) -> None:
    with pytest.raises(KeyError, match="unknown model"):
        run_experiment(config.model_copy(update={"model": ComponentSpec(name="nope")}))
    with pytest.raises(KeyError, match="unknown probe"):
        run_experiment(config.model_copy(update={"probes": ("nope",)}))
    assert not Path(config.output_dir).exists()


def test_compare_identical_configs(config: ExperimentConfig) -> None:
    a, b = run_experiment(config), run_experiment(config)
    result = compare_runs(a, b)
    assert [r["nfe"] for r in result.rows] == [2, 4]
    assert all(r["audio_identical"] == "9/9" for r in result.rows)
    assert result.warnings == []

    fewer_repeats = config.measurement.model_copy(update={"repeats": 2})
    other = run_experiment(config.model_copy(update={"measurement": fewer_repeats}))
    assert any("measurement settings" in w for w in compare_runs(a, other).warnings)


def test_cli_end_to_end(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = workspace / "c.yaml"
    cfg.write_text("name: cli\nmodel: {name: sine}\nsweep: {nfe: [2]}\nmeasurement: {warmup: 0, repeats: 2}\n")
    assert main(["run", str(cfg), "--set", "sweep.nfe=[1,3]"]) == 0
    (run,) = list_runs(workspace / "runs")
    assert run.manifest["config"]["sweep"]["nfe"] == [1, 3]

    assert main(["ls", str(workspace / "runs")]) == 0
    assert run.path.name in capsys.readouterr().out

    assert main(["compare", str(run.path), str(run.path)]) == 0
    assert main(["summarize", str(run.path)]) == 0
