import numpy as np

from synth import SR, clean
from wav_validation.checks import Thresholds, count_non_finite, is_too_short

def test_clean_audio_has_no_non_finite_samples():
    assert count_non_finite(clean()) == 0


def test_nan_and_inf_are_counted():
    audio = clean()
    audio[100] = np.nan
    audio[200] = np.inf
    audio[300] = -np.inf
    assert count_non_finite(audio) == 3


def test_too_short_boundary_is_half_a_second():
    assert is_too_short(SR // 2 - 1, SR)
    assert not is_too_short(SR // 2, SR)
    assert is_too_short(0, SR)


def test_too_short_respects_custom_threshold():
    assert is_too_short(SR, SR, Thresholds(min_seconds=2.0))
