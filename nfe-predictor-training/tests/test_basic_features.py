"""Tests for inexpensive text features."""

import pytest

from features.basic import extract_basic_features


def test_empty_text_returns_zero_features() -> None:
    assert extract_basic_features("") == {
        "character_count": 0.0,
        "word_count": 0.0,
        "punctuation_count": 0.0,
        "digit_count": 0.0,
        "average_word_length": 0.0,
    }


def test_ordinary_text() -> None:
    features = extract_basic_features("Hello world")

    assert features["character_count"] == 11.0
    assert features["word_count"] == 2.0
    assert features["punctuation_count"] == 0.0
    assert features["digit_count"] == 0.0
    assert features["average_word_length"] == 5.0


def test_punctuation_is_counted() -> None:
    features = extract_basic_features("Hello, world!")

    assert features["punctuation_count"] == 2.0
    assert features["average_word_length"] == 6.0


def test_individual_digits_are_counted() -> None:
    features = extract_basic_features("I have 2 cats and 10 dogs.")

    assert features["digit_count"] == 3.0
    assert features["punctuation_count"] == 1.0


def test_special_characters_are_counted() -> None:
    features = extract_basic_features("Wait... what?! #confused")

    assert features["punctuation_count"] == 6.0


def test_whitespace_only_text_has_no_words() -> None:
    features = extract_basic_features("   \n\t ")

    assert features["character_count"] == 6.0
    assert features["word_count"] == 0.0
    assert features["average_word_length"] == 0.0


def test_long_prompt() -> None:
    text = "word " * 10_000
    features = extract_basic_features(text)

    assert features["character_count"] == 50_000.0
    assert features["word_count"] == 10_000.0
    assert features["average_word_length"] == 4.0


@pytest.mark.parametrize("invalid_text", [None, 123, ["hello"]])
def test_non_string_input_raises_type_error(invalid_text: object) -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        extract_basic_features(invalid_text)  # type: ignore[arg-type]
