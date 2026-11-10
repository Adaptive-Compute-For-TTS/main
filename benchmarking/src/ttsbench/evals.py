"""Run a self-contained eval bundle (e.g. the WER/CER package) on a remote backend.

A bundle is a folder with a `requirements.txt` and an entrypoint. ttsbench ships it to Modal,
installs the requirements, runs the command from the bundle root, and copies the command's
output folder back. `{out}` in the command is replaced with the remote output folder:

    ttsbench eval path/to/bundle --out evals/wer-smoke -- \\
        python sample_eval/run_eval.py --manifest sample_eval/manifest.csv --out {out}

To score a ttsbench run, write its manifest and ship the run folder as data (mounted at /data):

    ttsbench manifest runs/<run>
    ttsbench eval path/to/bundle --data runs/<run> --out evals/<run> -- \\
        python sample_eval/run_eval.py --manifest /data/eval_manifest.csv --out {out}
"""

from __future__ import annotations

import csv
import io
import json
import subprocess
import tarfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Stdlib-only at import time: this module is imported inside eval containers, which only have the
# bundle's own requirements installed.
from ttsbench.measure.environment import worker_environment

if TYPE_CHECKING:
    from ttsbench.storage import Run

OUT_PLACEHOLDER = "{out}"
_REMOTE_BUNDLE = "/bundle"
_REMOTE_OUT = "/tmp/eval-out"
_REMOTE_DATA = "/data"
MANIFEST_NAME = "eval_manifest.csv"
MANIFEST_COLUMNS = ("clip_id", "audio_path", "reference_text", "run_id", "prompt_id", "nfe", "repeat", "length", "seed")
_IGNORE = ["__MACOSX", "**/__pycache__", "**/.DS_Store"]


def write_eval_manifest(run: Run, path: Path | None = None) -> Path:
    """One row per saved clip, in the eval manifest format (audio_path relative to the run folder)."""
    path = path or run.path / MANIFEST_NAME
    prompts = {p.id: p for p in run.prompts()}
    run_id = run.path.name
    rows = [
        {
            "clip_id": f"{s.prompt_id}__nfe{s.nfe}__r{s.repeat}",
            "audio_path": s.audio_path,
            "reference_text": prompts[s.prompt_id].text,
            "run_id": run_id,
            "prompt_id": s.prompt_id,
            "nfe": s.nfe,
            "repeat": s.repeat,
            "length": prompts[s.prompt_id].length,
            "seed": s.seed,
        }
        for s in run.samples()
        if s.audio_path is not None
    ]
    if not rows:
        raise ValueError(f"run {run_id} has no saved audio (measurement.save_audio was 'none')")
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def substitute_out(command: list[str], out: str) -> list[str]:
    if not any(OUT_PLACEHOLDER in part for part in command):
        raise ValueError(f"command must contain {OUT_PLACEHOLDER} so its outputs can be collected")
    return [part.replace(OUT_PLACEHOLDER, out) for part in command]


def tar_dir(path: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        if path.is_dir():
            tar.add(path, arcname=".")
    return buffer.getvalue()


def untar_into(data: bytes, dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")  # "data" filter rejects absolute paths and traversal
        return sorted(m.name.removeprefix("./") for m in tar.getmembers() if m.isfile())


def run_bundle_command(command: list[str]) -> dict[str, Any]:
    """Runs inside the container."""
    Path(_REMOTE_OUT).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    proc = subprocess.run(command, cwd=_REMOTE_BUNDLE, capture_output=True, text=True, check=False)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "wall_time_s": round(time.perf_counter() - t0, 2),
        "environment": worker_environment(),
        "out_tar": tar_dir(Path(_REMOTE_OUT)),
    }


def run_eval_on_modal(
    bundle: Path,
    command: list[str],
    out: Path,
    requirements: str = "requirements.txt",
    gpu: str | None = None,
    data: Path | None = None,
    python: str = "3.11",
    timeout_s: int = 1800,
    cache_volume: str = "ttsbench-hf-cache",
) -> dict[str, Any]:
    import modal

    remote_command = substitute_out(command, _REMOTE_OUT)
    req_path = bundle / requirements
    if not req_path.is_file():
        raise FileNotFoundError(f"bundle has no {requirements}: {req_path}")

    image = (
        modal.Image.debian_slim(python_version=python)
        .apt_install("ffmpeg")
        .pip_install_from_requirements(str(req_path))
        .env({"HF_HOME": "/cache/hf"})
        .add_local_dir(bundle, _REMOTE_BUNDLE, ignore=_IGNORE)
    )
    if data is not None:
        image = image.add_local_dir(data, _REMOTE_DATA, ignore=_IGNORE)
    image = image.add_local_python_source("ttsbench")
    app = modal.App("ttsbench-eval")
    fn = app.function(
        image=image,
        gpu=gpu,
        timeout=timeout_s,
        volumes={"/cache": modal.Volume.from_name(cache_volume, create_if_missing=True)},
        serialized=True,
    )(run_bundle_command)

    with modal.enable_output(), app.run():
        result = fn.remote(remote_command)

    files = untar_into(result.pop("out_tar"), out)
    record = {
        "bundle": bundle.name,
        "command": command,
        "data": str(data) if data else None,
        "backend": {"name": "modal", "gpu": gpu, "python": python},
        "files": files,
        **result,
    }
    (out / "eval_run.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    return record
