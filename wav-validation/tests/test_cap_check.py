from wav_validation.checks import cap_check


def test_hitting_the_patch_limit_is_at_cap():
    c = cap_check(12.8, num_patches=80, max_generate_length=80)
    assert c == {"cap_s": 12.8, "at_cap": True, "cap_source": "patches"}


def test_stopping_before_the_limit_is_not_at_cap():
    c = cap_check(9.6, num_patches=60, max_generate_length=80)
    assert c["at_cap"] is False and c["cap_source"] == "patches"


def test_patch_count_wins_over_duration():
    # Duration alone would look capped, but the executor says it stopped on its own.
    assert cap_check(12.8, num_patches=79, max_generate_length=80)["at_cap"] is False


def test_no_cap_known_means_no_answer():
    assert cap_check(12.8, num_patches=80, max_generate_length=None) == {
        "cap_s": None, "at_cap": None, "cap_source": None}


def test_default_benchmark_cap_of_500_patches_is_80_seconds():
    assert cap_check(10.0, num_patches=62, max_generate_length=500)["cap_s"] == 80.0


def test_duration_fallback_when_no_patch_count():
    c = cap_check(12.8, num_patches=None, max_generate_length=80)
    assert c == {"cap_s": 12.8, "at_cap": True, "cap_source": "duration"}


def test_duration_fallback_tolerance_is_10_ms():
    # Not exactly 12.79: 12.8 - 0.01 is 12.790000000000001 in floating point (same in the old script).
    assert cap_check(12.795, None, 80)["at_cap"] is True
    assert cap_check(12.785, None, 80)["at_cap"] is False


def test_neither_given_skips_the_check():
    assert cap_check(12.8, None, None)["at_cap"] is None
