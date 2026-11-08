from wav_validation.checks import CAUTION, FLAGS, INVALID


def test_every_flag_has_a_known_severity_and_description():
    for name, (severity, description) in FLAGS.items():
        assert severity in (INVALID, CAUTION), name
        assert description.strip(), name


def test_invalid_flags_come_before_cautions():
    severities = [severity for severity, _ in FLAGS.values()]
    assert severities == sorted(severities, key=lambda s: s != INVALID)


def test_names_shared_with_quality_evals_keep_their_meaning():
    # build_run_json.py uses these names with the same severity.
    for name in ("missing_audio", "unreadable_audio", "too_short", "silent", "truncated_at_cap"):
        assert FLAGS[name][0] == INVALID


def test_cautions_are_exactly_the_old_signal_ok_conditions():
    cautions = {name for name, (severity, _) in FLAGS.items() if severity == CAUTION}
    assert cautions == {"clipped", "too_loud", "leading_silence", "abrupt_end", "long_gap"}
