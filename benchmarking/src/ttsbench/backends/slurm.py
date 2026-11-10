"""Run jobs on a SLURM cluster over SSH (WATcloud by default via the `watcloud` preset).

Per job: write job.json + an sbatch script under `~/<remote_root>/jobs/<id>/`, submit, poll
sacct until the job ends, then read result.json back. The source tree is uploaded once on
enter. The Python env is built inside the job (on the GPU node), keyed by the model's pinned
requirements, so it is created once and reused.

Latency hygiene: the default GPU request is a whole GPU (shared "shard" GPUs make numbers
noisy), and after the first job every later job is pinned to the same node.
"""

from __future__ import annotations

import hashlib
import io
import shlex
import subprocess
import tarfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ttsbench.backends.base import Backend
from ttsbench.models import MODELS
from ttsbench.schema import ComponentSpec, Job, JobResult

TERMINAL_OK = {"COMPLETED"}
TERMINAL_FAILED = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
_BASE_PACKAGES = ("numpy>=1.26", "pydantic>=2.10", "pyyaml>=6.0", "soundfile>=0.12")
_UPLOAD = ("src/ttsbench", "benchmarks")

# (command, stdin) -> stdout. Swappable so tests can drive a fake cluster.
Shell = Callable[[str, bytes | None], bytes]


def ssh_shell(host: str, options: tuple[str, ...] = ()) -> Shell:
    def run(command: str, stdin: bytes | None = None) -> bytes:
        proc = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", *options, host, command], input=stdin, capture_output=True, check=False
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ssh {host} failed ({proc.returncode}): {command}\n{proc.stderr.decode()[-2000:]}")
        return proc.stdout

    return run


def _tarball(local_root: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for rel in _UPLOAD:
            tar.add(local_root / rel, arcname=rel, filter=lambda t: None if "__pycache__" in t.name else t)
    return buffer.getvalue()


class SlurmBackend(Backend):
    def __init__(
        self,
        model: ComponentSpec,
        host: str,
        partition: str | None = None,
        gres: str = "gpu:1",
        cpus: int = 8,
        mem: str = "32G",
        time_limit: str = "02:00:00",
        remote_root: str = "ttsbench",
        python_env: str = "uv",
        python: str = "python3",
        hf_home: str | None = None,
        ssh_options: tuple[str, ...] = (),
        poll_s: float = 10.0,
        local_root: str = ".",
        shell: Shell | None = None,
    ) -> None:
        super().__init__(model)
        if python_env not in ("uv", "existing"):
            raise ValueError("python_env must be 'uv' (build a pinned env) or 'existing' (use `python` as is)")
        self.host = host
        self.partition = partition
        self.gres = gres
        self.cpus = cpus
        self.mem = mem
        self.time_limit = time_limit
        self.remote_root = remote_root.strip("/")
        self.python_env = python_env
        self.python = python
        self.hf_home = hf_home
        self.poll_s = poll_s
        self.local_root = Path(local_root)
        self.shell = shell or ssh_shell(host, tuple(ssh_options))
        self.node: str | None = None
        self._active_job: str | None = None
        self._session = uuid.uuid4().hex[:8]

    # -- lifecycle -----------------------------------------------------------------------------

    def __enter__(self) -> SlurmBackend:
        root = self._root()
        self.shell(f"rm -rf {root}/src && mkdir -p {root} && tar -xzf - -C {root}", _tarball(self.local_root))
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self._active_job is not None:
            self.shell(f"scancel {self._active_job}", None)
            self._active_job = None

    # -- execution -----------------------------------------------------------------------------

    def execute(self, job: Job) -> JobResult:
        job_dir = f"{self._root()}/jobs/{self._session}-nfe{job.nfe:03d}"
        self.shell(f"mkdir -p {job_dir} && cat > {job_dir}/job.json", job.model_dump_json().encode())
        self.shell(f"cat > {job_dir}/job.sbatch", self.render_script(job, job_dir).encode())
        job_id = self.shell(f"sbatch --parsable {job_dir}/job.sbatch", None).decode().strip().split(";")[0]
        self._active_job = job_id
        state, node = self._wait(job_id, job_dir)
        self._active_job = None
        if state not in TERMINAL_OK:
            log = self.shell(f"tail -n 50 {job_dir}/slurm.out 2>/dev/null || true", None).decode()
            raise RuntimeError(f"SLURM job {job_id} ended {state}\n{log}")
        self.node = self.node or node
        return JobResult.model_validate_json(self.shell(f"cat {job_dir}/result.json", None))

    def _wait(self, job_id: str, job_dir: str) -> tuple[str, str | None]:
        while True:
            out = self.shell(f"sacct -j {job_id} -X -n -P -o State,NodeList 2>/dev/null || true", None).decode().strip()
            if out:
                state, _, node = out.splitlines()[0].partition("|")
                state = state.split()[0]  # "CANCELLED by 123" -> "CANCELLED"
                if state in TERMINAL_OK | TERMINAL_FAILED:
                    return state, node or None
            elif not self.shell(f"squeue -h -j {job_id} -o %T 2>/dev/null || true", None).strip():
                # No accounting record and no longer queued: judge by whether a result was written.
                done = self.shell(f"test -f {job_dir}/result.json && echo yes || true", None).strip()
                return ("COMPLETED" if done else "FAILED"), None
            time.sleep(self.poll_s)

    # -- script --------------------------------------------------------------------------------

    def _root(self) -> str:
        return f"$HOME/{self.remote_root}"

    def _env_setup(self) -> list[str]:
        if self.python_env == "existing":
            return [f"PY={shlex.quote(self.python)}"]
        req = MODELS.resolve(self.model.name).requirements
        packages = [*req.pip, *_BASE_PACKAGES]
        env_id = hashlib.sha256("\n".join([req.python, *packages]).encode()).hexdigest()[:12]
        pkgs = " ".join(shlex.quote(p) for p in packages)
        venv = f"{self._root()}/envs/{env_id}"
        return [
            'export PATH="$HOME/.local/bin:$PATH"',
            "command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh",
            f'if [ ! -x "{venv}/bin/python" ]; then',
            f'  uv venv --python {req.python} "{venv}.tmp" && uv pip install --python "{venv}.tmp/bin/python" {pkgs}',
            f'  rm -rf "{venv}" && mv "{venv}.tmp" "{venv}"',
            "fi",
            f'PY="{venv}/bin/python"',
        ]

    def render_script(self, job: Job, job_dir: str) -> str:
        directives = [
            f"--job-name=ttsbench-nfe{job.nfe}",
            f"--gres={self.gres}",
            f"--cpus-per-task={self.cpus}",
            f"--mem={self.mem}",
            f"--time={self.time_limit}",
            f"--output={job_dir}/slurm.out".replace("$HOME/", ""),
        ]
        if self.partition:
            directives.append(f"--partition={self.partition}")
        if self.node:
            directives.append(f"--nodelist={self.node}")
        env = [f'export TTSBENCH_ASSETS_ROOT="{self._root()}/benchmarks"', f'export PYTHONPATH="{self._root()}/src"']
        if self.hf_home:
            env.append(f"export HF_HOME={shlex.quote(self.hf_home)}")
        lines = [
            "#!/bin/bash",
            *(f"#SBATCH {d}" for d in directives),
            "set -euo pipefail",
            *self._env_setup(),
            *env,
            "nvidia-smi -L || true",
            f'"$PY" -m ttsbench.remote "{job_dir}/job.json" "{job_dir}/result.json"',
        ]
        return "\n".join(lines) + "\n"

    def describe(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "partition": self.partition,
            "gres": self.gres,
            "gpu": self.gres,
            "exclusive_gpu": not self.gres.startswith("shard"),
        }


def watcloud(model: ComponentSpec, **params: Any) -> SlurmBackend:
    """WATcloud preset (https://cloud.watonomous.ca/docs/compute-cluster/slurm)."""
    return SlurmBackend(model, **{"host": "wato-login1", "partition": "compute", **params})
