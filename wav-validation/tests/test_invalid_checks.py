import numpy as np

from synth import clean
from wav_validation.checks import count_non_finite


def test_clean_audio_has_no_non_finite_samples():
    assert count_non_finite(clean()) == 0


def test_nan_and_inf_are_counted():
    audio = clean()
    audio[100] = np.nan
    audio[200] = np.inf
    audio[300] = -np.inf
    assert count_non_finite(audio) == 3
