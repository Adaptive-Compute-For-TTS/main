import json

import numpy as np
import pytest

from synth import SR, clean, join, silence, tone
from wav_validation.checks import FLAGS, Thresholds, validate_audio


def test_clean_clip_passes():
    r = validate_audio(clean(), SR)
    assert r["ok"] is True
    assert r["flags"] == []
    assert r["signal_ok"] is True
    assert r["checks"]["duration_s"] == pytest.approx(1.2)
    assert r["checks"]["sample_rate"] == SR and r["checks"]["channels"] == 1


def test_result_is_json_serialisable_with_plain_types():
    r = validate_audio(clean(), SR, num_patches=8, max_generate_length=500)
    assert json.loads(json.dumps(r)) == r


def test_result_has_the_documented_keys():
    r = validate_audio(clean(), SR)
    assert set(r) == {"ok", "flags", "signal_ok", "checks", "thresholds", "version"}
    assert set(r["checks"]) == {
        "duration_s", "sample_rate", "channels", "peak", "rms", "non_finite_samples", "clipped_samples",
        "leading_silence_s", "trailing_silence_s", "longest_gap_s", "num_patches", "max_generate_length",
        "cap_s", "at_cap", "cap_source"}


@pytest.mark.parametrize("audio, flag", [
    (join(clean()[:1000], np.array([np.nan], dtype=np.float32), clean()), "non_finite"),
    (tone(0.2), "too_short"),
    (silence(1.0), "silent"),
])
def test_invalid_content_fails_and_skips_signal_measurements(audio, flag):
    r = validate_audio(audio, SR)
    assert r["ok"] is False
    assert r["flags"] == [flag]
    assert r["signal_ok"] is None
    assert r["checks"]["leading_silence_s"] is None


def test_truncated_clip_fails_but_still_gets_signal_checks():
    r = validate_audio(join(silence(0.1), tone(1.0)), SR, num_patches=80, max_generate_length=80)
    assert r["ok"] is False
    assert r["flags"] == ["truncated_at_cap", "abrupt_end"]
    assert r["signal_ok"] is False


def test_cautions_alone_do_not_fail_the_run():
    r = validate_audio(join(silence(0.7), tone(1.0, amp=0.97), silence(1.5), tone(0.3)), SR)
    assert r["ok"] is True
    assert r["flags"] == ["too_loud", "leading_silence", "abrupt_end", "long_gap"]
    assert r["signal_ok"] is False


def test_flags_are_in_table_order():
    r = validate_audio(join(silence(0.1), tone(1.0)), SR, num_patches=80, max_generate_length=80)
    order = list(FLAGS)
    assert r["flags"] == sorted(r["flags"], key=order.index)


def test_custom_thresholds_are_used_and_reported():
    t = Thresholds(min_seconds=2.0)
    r = validate_audio(clean(), SR, thresholds=t)
    assert r["flags"] == ["too_short"]
    assert r["thresholds"]["min_seconds"] == 2.0


@pytest.mark.parametrize("bad_sr", [0, -1, 1.5, True])
def test_bad_sample_rate_raises(bad_sr):
    with pytest.raises(ValueError):
        validate_audio(clean(), bad_sr)


def test_stereo_input_is_reported_as_two_channels():
    stereo = np.stack([clean(), clean()], axis=1)
    r = validate_audio(stereo, SR)
    assert r["ok"] and r["checks"]["channels"] == 2


def test_peak_is_reported_at_full_precision_so_reports_round_once():
    # Found on the archive: 0.15155029 rounded to 6 then 4 digits gave 0.1515, not 0.1516.
    audio = clean()
    audio[1000] = np.float32(0.1515502930)
    audio = audio * (0.1515502930 / np.abs(audio).max())
    r = validate_audio(audio.astype(np.float32), SR)
    assert round(r["checks"]["peak"], 4) == round(float(np.abs(audio.astype(np.float32)).max()), 4) == 0.1516
