"""Basic, inexpensive text features for NFE prediction."""

import string


def extract_basic_features(text: str) -> dict[str, float]:
    """Extract deterministic scalar features from ``text``.

    Words are whitespace-separated tokens. Character count includes
    whitespace, punctuation count uses Python's ASCII ``string.punctuation``
    set, and digit count counts individual numeric characters. Average word
    length includes every character in each token, including punctuation.

    Args:
        text: Prompt text to describe.

    Returns:
        A mapping containing numeric values for each basic feature.

    Raises:
        TypeError: If ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    words = text.split()
    word_count = len(words)
    average_word_length = (
        sum(len(word) for word in words) / word_count if word_count else 0.0
    )

    return {
        "character_count": float(len(text)),
        "word_count": float(word_count),
        "punctuation_count": float(
            sum(character in string.punctuation for character in text)
        ),
        "digit_count": float(sum(character.isdigit() for character in text)),
        "average_word_length": float(average_word_length),
    }
