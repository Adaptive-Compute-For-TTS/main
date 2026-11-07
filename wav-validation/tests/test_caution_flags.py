from synth import SR, clean, join, silence, tone
from wav_validation.checks import caution_flags, peak_and_rms, signal_measurements


def flags_for(audio):
    peak, _ = peak_and_rms(audio)
    return caution_flags(peak, signal_measurements(audio, SR))


def test_clean_clip_has_no_cautions():
    assert flags_for(clean()) == []


def test_full_scale_sample_is_clipped_and_too_loud():
    audio = clean()
    audio[5000] = 1.0
    assert flags_for(audio) == ["clipped", "too_loud"]


def test_loud_but_not_clipped():
    audio = join(silence(0.1), tone(1.0, amp=0.97), silence(0.1))
    assert flags_for(audio) == ["too_loud"]


def test_peak_exactly_at_limit_is_fine():
    assert caution_flags(0.95, signal_measurements(clean(), SR)) == []


def test_long_lead_in_is_flagged():
    audio = join(silence(0.7), tone(1.0), silence(0.1))
    assert flags_for(audio) == ["leading_silence"]


def test_lead_in_just_under_limit_is_fine():
    audio = join(silence(0.46), tone(1.0), silence(0.1))
    assert flags_for(audio) == []


def test_no_tail_is_an_abrupt_end():
    audio = join(silence(0.1), tone(1.0))
    assert flags_for(audio) == ["abrupt_end"]


def test_short_tail_of_two_frames_is_enough():
    audio = join(silence(0.1), tone(1.0), silence(0.06))
    assert flags_for(audio) == []
