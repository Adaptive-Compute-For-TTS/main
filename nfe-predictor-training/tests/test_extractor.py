"""Tests for the canonical feature extraction interface."""

import pytest

from features import FEATURE_NAMES, extract_features, features_to_vector


def test_extractor_returns_canonical_features() -> None:
    features = extract_features("Hi, 2 you!", context={"speaker_id": "demo"})

    assert tuple(features) == FEATURE_NAMES
    assert features == {
        "character_count": 10.0,
        "word_count": 3.0,
        "punctuation_count": 2.0,
        "digit_count": 1.0,
        "average_word_length": pytest.approx(8 / 3),
    }


def test_features_to_vector_uses_canonical_order() -> None:
    unordered_features = {
        "average_word_length": 5.5,
        "digit_count": 4.0,
        "punctuation_count": 3.0,
        "word_count": 2.0,
        "character_count": 1.0,
    }

    assert features_to_vector(unordered_features) == [1.0, 2.0, 3.0, 4.0, 5.5]


def test_features_to_vector_rejects_missing_feature() -> None:
    with pytest.raises(KeyError, match="word_count"):
        features_to_vector({"character_count": 1.0})
