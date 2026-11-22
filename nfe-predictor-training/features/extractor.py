"""Canonical feature extraction entry point for NFE prediction."""

from collections.abc import Mapping
from typing import Any, Optional

from features.basic import extract_basic_features


FEATURE_NAMES = (
    "character_count",
    "word_count",
    "punctuation_count",
    "digit_count",
    "average_word_length",
)


def extract_features(
    text: str,
    context: Optional[Mapping[str, Any]] = None,
) -> dict[str, float]:
    """Extract all enabled pre-generation features for a prompt.

    ``context`` is reserved for later speaker metadata or embedding features.
    It is accepted now so training and inference can share this stable API.
    """
    del context
    return extract_basic_features(text)


def features_to_vector(features: Mapping[str, float]) -> list[float]:
    """Convert a feature mapping to the canonical model-input order."""
    return [float(features[name]) for name in FEATURE_NAMES]
