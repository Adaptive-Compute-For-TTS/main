import numpy as np
import pytest

from synth import SR, clean, join, silence, tone
from wav_validation.checks import signal_measurements


def test_clean_clip_layout():
    m = signal_measurements(clean(), SR)  # 0.1 s silence, 1 s tone, 0.1 s silence
    assert m["clipped_samples"] == 0
    assert m["leading_silence_s"] == pytest.approx(0.1)
    assert m["trailing_silence_s"] == pytest.approx(0.1)
    assert m["longest_gap_s"] == 0.0


def test_gap_in_the_middle_is_measured():
    audio = join(tone(0.5), silence(1.5), tone(0.5), silence(0.1))
    assert signal_measurements(audio, SR)["longest_gap_s"] == pytest.approx(1.5)


def test_full_scale_samples_are_counted_as_clipped():
    audio = clean()
    audio[1000:1010] = 1.0
    audio[2000] = -1.0
    assert signal_measurements(audio, SR)["clipped_samples"] == 11


def test_audio_shorter_than_one_frame_does_not_crash():
    m = signal_measurements(tone(0.01), SR)  # 10 ms < one 20 ms frame
    assert m == {"clipped_samples": 0, "leading_silence_s": 0.0, "trailing_silence_s": 0.0,
                 "longest_gap_s": 0.0}


def test_empty_audio_does_not_crash():
    assert signal_measurements(np.zeros(0, dtype=np.float32), SR)["longest_gap_s"] == 0.0


def test_returns_plain_python_numbers():
    for value in signal_measurements(clean(), SR).values():
        assert type(value) in (int, float)


def test_matches_old_script_signal_checks(tmp_path):
    """Same numbers as flag_wav_runs.py:signal_checks on the same clip."""
    import importlib.util
    from pathlib import Path

    from synth import write

    old = Path(__file__).resolve().parents[1] / "flag_wav_runs.py"
    spec = importlib.util.spec_from_file_location("flag_wav_runs", old)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    audio = join(silence(0.3), tone(0.7, amp=0.4), silence(1.2), tone(0.4, amp=0.9), silence(0.04))
    path = write(tmp_path / "x.wav", audio)
    expected = module.signal_checks(path)
    got = signal_measurements(audio, SR)
    assert got["leading_silence_s"] == expected["leading_silence_seconds"]
    assert got["trailing_silence_s"] == expected["trailing_silence_seconds"]
    assert got["longest_gap_s"] == expected["longest_internal_gap_seconds"]
    assert got["clipped_samples"] == expected["clipped_samples"]
