import numpy as np

from synth import SR, clean, join, silence, tone, write
from wav_validation.checks import validate_audio, validate_wav


def test_file_gives_same_result_as_in_memory(tmp_path):
    audio = clean()
    assert validate_wav(write(tmp_path / "a.wav", audio)) == validate_audio(audio, SR)


def test_missing_file(tmp_path):
    r = validate_wav(tmp_path / "nope.wav")
    assert r["ok"] is False and r["flags"] == ["missing_audio"]
    assert r["checks"]["duration_s"] is None


def test_garbage_bytes_are_unreadable(tmp_path):
    path = tmp_path / "junk.wav"
    path.write_bytes(b"this is not a wav file" * 10)
    r = validate_wav(path)
    assert r["ok"] is False and r["flags"] == ["unreadable_audio"]


def test_directory_is_unreadable(tmp_path):
    assert validate_wav(tmp_path)["flags"] == ["unreadable_audio"]


def test_file_problem_has_same_keys_as_a_measured_result(tmp_path):
    measured = validate_audio(clean(), SR)
    missing = validate_wav(tmp_path / "nope.wav", num_patches=3, max_generate_length=80)
    assert set(missing) == set(measured)
    assert set(missing["checks"]) == set(measured["checks"])
    assert missing["checks"]["num_patches"] == 3


def test_int16_pcm_is_read(tmp_path):
    path = write(tmp_path / "pcm.wav", clean(), subtype="PCM_16")
    r = validate_wav(path)
    assert r["ok"] and r["flags"] == []


def test_stereo_file(tmp_path):
    stereo = np.stack([clean(), clean()], axis=1)
    path = write(tmp_path / "stereo.wav", stereo)
    assert validate_wav(path)["checks"]["channels"] == 2


def test_nan_survives_a_float_wav_and_is_caught(tmp_path):
    audio = clean()
    audio[500] = np.nan
    assert validate_wav(write(tmp_path / "nan.wav", audio))["flags"] == ["non_finite"]


def test_duration_fallback_on_a_file_at_the_cap(tmp_path):
    audio = join(silence(0.1), tone(12.7))  # 12.8 s = 80 patches
    r = validate_wav(write(tmp_path / "capped.wav", audio), max_generate_length=80)
    assert "truncated_at_cap" in r["flags"] and r["checks"]["cap_source"] == "duration"
