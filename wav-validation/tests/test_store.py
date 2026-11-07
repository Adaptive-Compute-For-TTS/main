import uuid

import pytest

from wav_validation.store import check_run_id


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
