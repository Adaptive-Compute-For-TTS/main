import subprocess
import sys

from synth import clean, join, silence, tone, write
from wav_validation.__main__ import main


def test_good_file_prints_ok_and_exits_0(tmp_path, capsys):
    path = write(tmp_path / "good.wav", clean())
    assert main([str(path)]) == 0
    assert capsys.readouterr().out.strip() == f"OK    {path}  -"


def test_bad_file_prints_fail_with_flags_and_exits_1(tmp_path, capsys):
    path = write(tmp_path / "quiet.wav", silence(1.0))
    assert main([str(path)]) == 1
    assert capsys.readouterr().out.strip() == f"FAIL  {path}  silent"


def test_cautions_are_shown_but_still_ok(tmp_path, capsys):
    path = write(tmp_path / "abrupt.wav", join(silence(0.1), tone(1.0)))
    assert main([str(path)]) == 0
    assert capsys.readouterr().out.strip().endswith("abrupt_end")


def test_any_failure_among_several_files_exits_1(tmp_path):
    good = write(tmp_path / "good.wav", clean())
    assert main([str(good), str(tmp_path / "missing.wav")]) == 1


def test_runs_as_python_dash_m(tmp_path):
    path = write(tmp_path / "good.wav", clean())
    out = subprocess.run([sys.executable, "-m", "wav_validation", str(path)], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.startswith("OK")
