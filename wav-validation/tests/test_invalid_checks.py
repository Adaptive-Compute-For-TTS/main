import numpy as np

from synth import SR, clean, silence, tone
from wav_validation.checks import Thresholds, count_non_finite, is_silent, is_too_short, peak_and_rms

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


def test_peak_and_rms_of_a_sine():
    peak, rms = peak_and_rms(tone(1.0, amp=0.5))
    assert abs(peak - 0.5) < 1e-3
    assert abs(rms - 0.5 / np.sqrt(2)) < 1e-3


def test_peak_and_rms_of_empty_audio():
    assert peak_and_rms(np.zeros(0, dtype=np.float32)) == (0.0, 0.0)


def test_digital_silence_is_silent():
    assert is_silent(*peak_and_rms(silence(1.0)))


def test_very_quiet_tone_is_silent_but_normal_speech_level_is_not():
    assert is_silent(*peak_and_rms(tone(1.0, amp=0.005)))  # peak below 0.01
    assert not is_silent(*peak_and_rms(clean()))


def test_single_click_in_silence_is_judged_by_rms():
    # Both clicks pass the peak rule (>= 0.01); the RMS rule decides.
    audio = silence(1.0)
    audio[0] = 0.05  # RMS = 0.05 / sqrt(48000) ~ 2.3e-4 >= 1e-4: not silent
    assert not is_silent(*peak_and_rms(audio))
    audio[0] = 0.02  # RMS ~ 9.1e-5 < 1e-4: silent
    assert is_silent(*peak_and_rms(audio))
