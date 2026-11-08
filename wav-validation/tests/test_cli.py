import json
import subprocess
import sys

from synth import clean, join, silence, tone, write
from wav_validation.__main__ import main


def test_good_file_prints_ok_and_exits_0(tmp_path, capsys):
    path = write(tmp_path / "good.wav", clean())
    assert main([str(path)]) == 0
    assert capsys.readouterr().out.splitlines()[0] == f"OK    {path}  -"


def test_bad_file_prints_fail_with_flags_and_exits_1(tmp_path, capsys):
    path = write(tmp_path / "quiet.wav", silence(1.0))
    assert main([str(path)]) == 1
    assert capsys.readouterr().out.splitlines()[0] == f"FAIL  {path}  silent"


def test_cautions_are_shown_but_still_ok(tmp_path, capsys):
    path = write(tmp_path / "abrupt.wav", join(silence(0.1), tone(1.0)))
    assert main([str(path)]) == 0
    assert capsys.readouterr().out.splitlines()[0].endswith("abrupt_end")


def test_any_failure_among_several_files_exits_1(tmp_path):
    good = write(tmp_path / "good.wav", clean())
    assert main([str(good), str(tmp_path / "missing.wav")]) == 1


def test_runs_as_python_dash_m(tmp_path):
    path = write(tmp_path / "good.wav", clean())
    out = subprocess.run([sys.executable, "-m", "wav_validation", str(path)], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.startswith("OK")


def test_directory_is_searched_recursively_and_summarised(tmp_path, capsys):
    write(tmp_path / "run" / "a" / "audio.wav", clean())
    write(tmp_path / "run" / "b" / "audio.wav", silence(1.0))
    write(tmp_path / "run" / "b" / "upper.WAV", clean())
    (tmp_path / "run" / "notes.txt").write_text("not audio")
    assert main([str(tmp_path / "run")]) == 1
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 4
    assert out[-1] == "3 files: 2 ok, 1 failed (silent 1)"


def test_empty_directory_is_an_error(tmp_path, capsys):
    assert main([str(tmp_path)]) == 1
    assert "no .wav files" in capsys.readouterr().err


def test_max_generate_length_turns_on_the_cut_off_check(tmp_path, capsys):
    path = write(tmp_path / "capped.wav", join(silence(0.1), tone(12.7)))  # 12.8 s = 80 patches
    assert main([str(path)]) == 0  # no cap given: can't tell
    assert main([str(path), "--max-generate-length", "80"]) == 1
    assert "truncated_at_cap" in capsys.readouterr().out.splitlines()[-2]


def test_json_output_is_parseable_and_summary_goes_to_stderr(tmp_path, capsys):
    good = write(tmp_path / "good.wav", clean())
    bad = write(tmp_path / "quiet.wav", silence(1.0))
    assert main([str(good), str(bad), "--json"]) == 1
    captured = capsys.readouterr()
    results = json.loads(captured.out)
    assert [r["path"] for r in results] == [str(good), str(bad)]
    assert [r["ok"] for r in results] == [True, False]
    assert results[1]["flags"] == ["silent"]
    assert captured.err.strip() == "2 files: 1 ok, 1 failed (silent 1)"
