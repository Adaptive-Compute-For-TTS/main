import uuid

import numpy as np
import pytest
import soundfile as sf

from synth import SR, clean
from wav_validation.store import check_run_id, write_wav_atomic


def test_backend_uuid_run_ids_are_accepted():
    run_id = str(uuid.uuid4())
    assert check_run_id(run_id) == run_id


@pytest.mark.parametrize("run_id", ["p004_nfe10", "run-1", "20261107.1"])
def test_simple_names_are_accepted(run_id):
    assert check_run_id(run_id) == run_id


@pytest.mark.parametrize("run_id", ["", ".", "..", "a/b", "../etc", "a\\b", "_rejected", ".hidden",
                                    "a..b", "x" * 200, None, 42, "with space"])
def test_unsafe_run_ids_are_refused(run_id):
    with pytest.raises(ValueError):
        check_run_id(run_id)


def test_atomic_write_reads_back_sample_identical_float32(tmp_path):
    audio = clean()
    path = write_wav_atomic(tmp_path / "run" / "audio.wav", audio, SR)
    back, sr = sf.read(path, dtype="float32")
    assert sr == SR and sf.info(path).subtype == "FLOAT"
    np.testing.assert_array_equal(back, audio)


def test_atomic_write_leaves_no_temp_files(tmp_path):
    write_wav_atomic(tmp_path / "audio.wav", clean(), SR)
    assert [p.name for p in tmp_path.iterdir()] == ["audio.wav"]


def test_failed_write_cleans_up_and_leaves_no_file(tmp_path):
    with pytest.raises(Exception):
        write_wav_atomic(tmp_path / "audio.wav", clean(), 0)  # invalid sample rate
    assert list(tmp_path.iterdir()) == []


def test_nan_audio_can_still_be_kept_for_debugging(tmp_path):
    audio = clean()
    audio[10] = np.nan
    back, _ = sf.read(write_wav_atomic(tmp_path / "bad.wav", audio, SR), dtype="float32")
    assert np.isnan(back[10])
