from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from ttsbench.evals import substitute_out, tar_dir, untar_into


def test_substitute_out_requires_placeholder() -> None:
    assert substitute_out(["python", "e.py", "--out", "{out}"], "/tmp/o") == ["python", "e.py", "--out", "/tmp/o"]
    assert substitute_out(["e.py", "--out={out}/x"], "/o") == ["e.py", "--out=/o/x"]
    with pytest.raises(ValueError, match="{out}"):
        substitute_out(["python", "e.py"], "/o")


def test_output_folder_round_trips(tmp_path: Path) -> None:
    src = tmp_path / "src"
    (src / "nested").mkdir(parents=True)
    (src / "results.json").write_text("{}")
    (src / "nested" / "per_clip.csv").write_text("a,b\n")
    files = untar_into(tar_dir(src), tmp_path / "dest")
    assert files == ["nested/per_clip.csv", "results.json"]
    assert (tmp_path / "dest" / "results.json").read_text() == "{}"
    assert untar_into(tar_dir(tmp_path / "missing"), tmp_path / "empty") == []


def test_untar_rejects_path_traversal(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(tarfile.TarError):
        untar_into(buffer.getvalue(), tmp_path / "dest")
    assert not (tmp_path / "escape.txt").exists()


def test_cli_passes_eval_command_through(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ttsbench import cli, evals

    seen = {}

    def fake_run(bundle, command, out, requirements, gpu, data):
        seen.update(bundle=bundle, command=command, out=out)
        return {"stdout": "", "stderr": "", "returncode": 0, "wall_time_s": 0.0}

    monkeypatch.setattr(evals, "run_eval_on_modal", fake_run)
    argv = ["eval", str(tmp_path), "--out", "o", "--", "python", "e.py", "--out", "{out}", "--check", "x.json"]
    assert cli.main(argv) == 0
    assert seen["command"] == ["python", "e.py", "--out", "{out}", "--check", "x.json"]
    assert seen["out"] == Path("o")


def test_write_eval_manifest(config) -> None:
    import csv as _csv

    from ttsbench.evals import write_eval_manifest
    from ttsbench.runner import run_experiment

    run = run_experiment(config)
    rows = list(_csv.DictReader(write_eval_manifest(run).open()))
    assert len(rows) == 2 * 3  # 2 NFE x 3 prompts, first repeat only (save_audio: first)
    first = rows[0]
    assert first["clip_id"] == "s1__nfe2__r0" and first["reference_text"] == "Short one."
    assert (run.path / first["audio_path"]).is_file()
    assert {r["nfe"] for r in rows} == {"2", "4"}


def test_evals_module_imports_without_third_party_packages() -> None:
    import subprocess
    import sys

    # Simulate an eval container: block pydantic/numpy/yaml imports, then import ttsbench.evals.
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in {'pydantic', 'numpy', 'yaml', 'soundfile'}:\n"
        "            raise ImportError(name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import ttsbench.evals\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
