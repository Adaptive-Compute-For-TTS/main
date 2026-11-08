"""The defaults must stay equal to the numbers the rest of the repo already uses."""

import ast
import dataclasses
from pathlib import Path

import pytest

from wav_validation.checks import DEFAULT, Thresholds

HERE = Path(__file__).resolve().parent
SPEAKER_SIM = HERE.parents[1] / "quality-evals" / "speaker_sim" / "speaker_sim.py"
OLD_SCRIPT = HERE.parent / "flag_wav_runs.py"


def module_constants(path: Path) -> dict:
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


@pytest.mark.skipif(not SPEAKER_SIM.exists(), reason="quality-evals not checked out")
def test_matches_speaker_sim_scorer():
    k = module_constants(SPEAKER_SIM)
    assert DEFAULT.min_seconds == k["MIN_SECONDS"]
    assert DEFAULT.min_peak == k["MIN_PEAK"]


def test_matches_old_benchmark_script():
    k = module_constants(OLD_SCRIPT)
    assert DEFAULT.frame_s == k["FRAME_SECONDS"]
    assert DEFAULT.silence_below_peak_db == k["SILENCE_BELOW_PEAK_DB"]
    assert DEFAULT.patch_seconds == k["PATCH_SECONDS"]
    assert DEFAULT.cap_tolerance_s == k["CAP_TOLERANCE_SECONDS"]
    # signal_ok() in the old script uses these as literals.
    src = OLD_SCRIPT.read_text()
    for literal in ('>= 0.999', '<= 0.95', '< 0.5', '>= 0.05', '<= 1.0'):
        assert literal in src
    assert (DEFAULT.clip_level, DEFAULT.max_peak, DEFAULT.max_leading_silence_s,
            DEFAULT.min_trailing_silence_s, DEFAULT.max_internal_gap_s) == (0.999, 0.95, 0.5, 0.05, 1.0)


def test_frozen_and_overridable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT.max_peak = 1.0
    custom = Thresholds(max_peak=0.9)
    assert custom.max_peak == 0.9 and DEFAULT.max_peak == 0.95
    assert custom.as_dict()["max_peak"] == 0.9
