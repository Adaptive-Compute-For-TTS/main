from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ttsbench.backends import BACKENDS
from ttsbench.backends.slurm import SlurmBackend
from ttsbench.schema import ComponentSpec, Job, MeasurementConfig, Prompt

REPO = Path(__file__).parents[1]

# Minimal SLURM stand-ins: sbatch runs the script synchronously and remembers its exit code.
SHIMS = {
    "sbatch": """#!/bin/bash
script="${@: -1}"
out=$(grep -m1 '^#SBATCH --output=' "$script" | cut -d= -f2-)
cp "$script" "$HOME/.last_script"
(cd "$HOME" && bash "$script" > "$HOME/$out" 2>&1); echo $? > "$HOME/.last_rc"
echo "4242;cluster"
""",
    "sacct": """#!/bin/bash
if [ "$(cat "$HOME/.last_rc")" = 0 ]; then echo "COMPLETED|gpu-node-7"; else echo "FAILED|gpu-node-7"; fi
""",
    "squeue": "#!/bin/bash\n",
    "scancel": '#!/bin/bash\necho cancelled >> "$HOME/.cancelled"\n',
}


@pytest.fixture
def fake_cluster(tmp_path: Path):
    home, bin_dir = tmp_path / "home", tmp_path / "bin"
    home.mkdir()
    bin_dir.mkdir()
    for name, body in SHIMS.items():
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    env = {**os.environ, "HOME": str(home), "PATH": f"{bin_dir}:{os.environ['PATH']}"}

    def shell(command: str, stdin: bytes | None = None) -> bytes:
        proc = subprocess.run(["bash", "-c", command], input=stdin, capture_output=True, env=env, cwd=home)
        assert proc.returncode == 0, proc.stderr.decode()
        return proc.stdout

    return home, shell


def _backend(shell, **params) -> SlurmBackend:
    return SlurmBackend(
        ComponentSpec(name="sine"),
        host="fake",
        python_env="existing",
        python=sys.executable,
        local_root=str(REPO),
        poll_s=0,
        shell=shell,
        **params,
    )


def _job(nfe: int, prompts: tuple[Prompt, ...]) -> Job:
    measurement = MeasurementConfig(warmup=1, repeats=2)
    return Job(model=ComponentSpec(name="sine"), nfe=nfe, prompts=prompts, measurement=measurement)


def test_jobs_round_trip_and_pin_to_first_node(fake_cluster, prompts: list[Prompt]) -> None:
    home, shell = fake_cluster
    with _backend(shell, gres="gpu:rtx_3090:1") as backend:
        first = backend.execute(_job(2, tuple(prompts[:2])))
        assert "--nodelist" not in (home / ".last_script").read_text()
        second = backend.execute(_job(4, tuple(prompts[:2])))

    assert backend.node == "gpu-node-7"
    script = (home / ".last_script").read_text()
    assert "#SBATCH --nodelist=gpu-node-7" in script and "#SBATCH --gres=gpu:rtx_3090:1" in script
    assert [len(first.samples), len(second.samples)] == [4, 4]
    assert first.audio and all(isinstance(v, bytes) for v in first.audio.values())
    assert second.samples[0].stage_calls["flow_step"] == 4


def test_failed_job_surfaces_the_remote_log(fake_cluster, prompts: list[Prompt]) -> None:
    _, shell = fake_cluster
    broken = prompts[0].model_copy(update={"reference_audio": "references/missing.wav"})
    with _backend(shell) as backend, pytest.raises(RuntimeError, match="(?s)FAILED.*missing.wav"):
        backend.execute(_job(2, (broken,)))


def test_uv_env_script_pins_model_requirements() -> None:
    backend = BACKENDS.create("watcloud", model=ComponentSpec(name="dots_tts"), shell=lambda c, s=None: b"")
    job = _job(16, ())
    script = backend.render_script(job, "$HOME/ttsbench/jobs/x")
    assert "#SBATCH --partition=compute" in script and "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --output=ttsbench/jobs/x/slurm.out" in script
    assert "uv venv --python 3.11" in script and " torch==2.8.0 " in script
    assert "'dots.tts @ git+https://github.com/studio-dots-ai/dots.tts.git@" in script
    assert backend.describe()["exclusive_gpu"] is True
    noop = lambda c, s=None: b""  # noqa: E731
    shard = BACKENDS.create("watcloud", model=ComponentSpec(name="sine"), gres="shard:16384", shell=noop)
    assert shard.describe()["exclusive_gpu"] is False


def test_invalid_python_env() -> None:
    with pytest.raises(ValueError):
        SlurmBackend(ComponentSpec(name="sine"), host="x", python_env="conda", shell=lambda c, s=None: b"")


def test_model_sources_are_uploaded_and_setup_runs_in_the_env_build(fake_cluster) -> None:
    home, shell = fake_cluster
    backend = SlurmBackend(
        ComponentSpec(name="dots_tts_executor"), host="fake", local_root=str(REPO), poll_s=0, shell=shell
    )
    with backend:
        pass
    assert (home / "ttsbench/build/gpu-executor/patches/dots_tts_scheduler_hook.patch").is_file()
    assert (home / "ttsbench/build/gpu-executor/src/adaptive_nfe/fixed_nfe.py").is_file()
    assert not list((home / "ttsbench/build").rglob("__pycache__"))

    script = backend.render_script(_job(4, ()), "$HOME/ttsbench/jobs/x")
    assert 'git -C "$HOME/ttsbench/build/dots.tts" apply "$HOME/ttsbench/build/gpu-executor/patches/' in script
    assert f".tmp-{backend._session}/bin/python" in script
    assert 'uv pip install --python "$HOME/ttsbench/envs/' in script
    assert script.index('dots.tts" apply') < script.index('  rm -rf "$HOME/ttsbench/envs/')


def test_env_id_changes_when_setup_changes() -> None:
    noop = lambda c, s=None: b""  # noqa: E731
    stock = SlurmBackend(ComponentSpec(name="dots_tts"), host="x", shell=noop)._env_setup()
    patched = SlurmBackend(ComponentSpec(name="dots_tts_executor"), host="x", shell=noop)._env_setup()
    assert stock[-1] != patched[-1]
