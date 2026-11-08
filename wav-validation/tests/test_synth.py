import numpy as np
import soundfile as sf

from synth import SR, clean, silence, tone, write


def test_helpers_build_expected_lengths():
    assert len(tone(1.0)) == SR
    assert len(silence(0.25)) == SR // 4
    assert len(clean(1.0)) == int(1.2 * SR)


def test_write_round_trips_float32(tmp_path):
    audio = clean()
    path = write(tmp_path / "a.wav", audio)
    back, sr = sf.read(path, dtype="float32")
    assert sr == SR
    np.testing.assert_array_equal(back, audio)
