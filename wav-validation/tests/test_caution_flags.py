from synth import SR, clean, join, silence, tone
from wav_validation.checks import caution_flags, is_signal_ok, peak_and_rms, signal_measurements


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


def test_pause_over_a_second_is_a_long_gap():
    audio = join(silence(0.1), tone(0.5), silence(1.5), tone(0.5), silence(0.1))
    assert flags_for(audio) == ["long_gap"]


def test_pause_of_exactly_one_second_is_fine():
    audio = join(silence(0.1), tone(0.5), silence(1.0), tone(0.5), silence(0.1))
    assert flags_for(audio) == []


def test_several_problems_are_all_reported_in_table_order():
    audio = join(silence(0.7), tone(0.5, amp=0.97), silence(1.5), tone(0.5))
    assert flags_for(audio) == ["too_loud", "leading_silence", "abrupt_end", "long_gap"]


def test_signal_ok_ignores_invalid_flags_and_fails_on_any_caution():
    assert is_signal_ok([])
    assert is_signal_ok(["truncated_at_cap"])
    assert not is_signal_ok(["abrupt_end"])
