import numpy as np
import pytest

from synth import tone
from wav_validation.checks import to_mono


def test_1d_passes_through_as_float32():
    audio = tone(0.1).astype(np.float64)
    mono, channels = to_mono(audio)
    assert channels == 1 and mono.dtype == np.float32 and mono.ndim == 1
    np.testing.assert_array_equal(mono, audio.astype(np.float32))


def test_single_channel_column_is_unchanged():
    audio = tone(0.1)
    mono, channels = to_mono(audio[:, None])
    assert channels == 1
    np.testing.assert_array_equal(mono, audio)


def test_stereo_is_averaged():
    left, right = tone(0.1, amp=0.2), tone(0.1, amp=0.6)
    mono, channels = to_mono(np.stack([left, right], axis=1))
    assert channels == 2
    np.testing.assert_allclose(mono, (left + right) / 2, atol=1e-7)


def test_plain_lists_are_accepted():
    mono, channels = to_mono([0.0, 0.5, -0.5])
    assert channels == 1 and mono.tolist() == [0.0, 0.5, -0.5]


@pytest.mark.parametrize("shape", [(1, 4800), (2, 4800), (2, 2, 10)])
def test_caller_bugs_raise(shape):
    with pytest.raises(ValueError):
        to_mono(np.zeros(shape, dtype=np.float32))
