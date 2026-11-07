import json
import uuid

import numpy as np
import pytest
import soundfile as sf

from synth import SR, clean, silence
from wav_validation.store import check_run_id, validate_and_store, write_wav_atomic


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


def test_passing_clip_is_stored_under_its_run_id(tmp_path):
    audio = clean()
    r = validate_and_store(audio, SR, "run-1", tmp_path)
    assert r["ok"] is True
    assert r["audio_path"] == str((tmp_path / "run-1" / "audio.wav").resolve())
    assert r["stored_path"] == r["audio_path"]
    back, _ = sf.read(r["audio_path"], dtype="float32")
    np.testing.assert_array_equal(back, audio)


def test_result_keeps_the_validation_fields(tmp_path):
    r = validate_and_store(clean(), SR, "run-2", tmp_path, num_patches=8, max_generate_length=500)
    assert r["flags"] == [] and r["checks"]["num_patches"] == 8 and r["checks"]["at_cap"] is False


def test_failing_clip_is_kept_under_rejected_with_no_audio_path(tmp_path):
    r = validate_and_store(silence(1.0), SR, "run-3", tmp_path)
    assert r["ok"] is False and r["audio_path"] is None
    assert r["stored_path"] == str((tmp_path / "_rejected" / "run-3" / "audio.wav").resolve())
    assert sf.info(r["stored_path"]).frames == SR
    assert not (tmp_path / "run-3").exists()  # nothing where the frontend would look


def test_cautions_alone_still_store_as_a_normal_result(tmp_path):
    from synth import join, tone
    r = validate_and_store(join(silence(0.1), tone(1.0)), SR, "run-4", tmp_path)  # abrupt_end only
    assert r["ok"] is True and r["flags"] == ["abrupt_end"]
    assert r["audio_path"] == str((tmp_path / "run-4" / "audio.wav").resolve())


def test_unsafe_run_id_is_refused_before_anything_is_written(tmp_path):
    with pytest.raises(ValueError):
        validate_and_store(clean(), SR, "../escape", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_validation_json_is_written_next_to_the_wav(tmp_path):
    r = validate_and_store(clean(), SR, "run-5", tmp_path)
    saved = json.loads((tmp_path / "run-5" / "validation.json").read_text())
    assert saved == r and saved["run_id"] == "run-5"


def test_rejected_clip_also_gets_validation_json(tmp_path):
    validate_and_store(silence(1.0), SR, "run-6", tmp_path)
    saved = json.loads((tmp_path / "_rejected" / "run-6" / "validation.json").read_text())
    assert saved["flags"] == ["silent"] and saved["audio_path"] is None


def test_run_folder_has_only_the_two_files(tmp_path):
    validate_and_store(clean(), SR, "run-7", tmp_path)
    assert sorted(p.name for p in (tmp_path / "run-7").iterdir()) == ["audio.wav", "validation.json"]


def test_storing_the_same_run_twice_is_refused(tmp_path):
    validate_and_store(clean(), SR, "run-8", tmp_path)
    with pytest.raises(FileExistsError):
        validate_and_store(clean(), SR, "run-8", tmp_path)


def test_a_rejected_run_cannot_be_stored_again_as_passed(tmp_path):
    validate_and_store(silence(1.0), SR, "run-9", tmp_path)
    with pytest.raises(FileExistsError):
        validate_and_store(clean(), SR, "run-9", tmp_path)
    assert not (tmp_path / "run-9").exists()


def test_refused_overwrite_leaves_the_original_untouched(tmp_path):
    first = validate_and_store(clean(), SR, "run-10", tmp_path)
    before = (tmp_path / "run-10" / "audio.wav").read_bytes()
    with pytest.raises(FileExistsError):
        validate_and_store(clean() * 0.5, SR, "run-10", tmp_path)
    assert (tmp_path / "run-10" / "audio.wav").read_bytes() == before
    assert json.loads((tmp_path / "run-10" / "validation.json").read_text()) == first


def test_stored_files_get_normal_permissions_not_owner_only(tmp_path):
    # Found in review: tempfile.mkstemp made audio.wav / validation.json mode 0600.
    import stat

    reference = tmp_path / "plain.txt"
    reference.write_text("x")  # whatever the current umask gives a normal file
    expected = stat.S_IMODE(reference.stat().st_mode)
    validate_and_store(clean(), SR, "perm", tmp_path)
    for name in ("audio.wav", "validation.json"):
        assert stat.S_IMODE((tmp_path / "perm" / name).stat().st_mode) == expected, name
