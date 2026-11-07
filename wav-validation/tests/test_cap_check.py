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
