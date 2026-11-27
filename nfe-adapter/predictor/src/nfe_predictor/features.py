"""Feature extractors, by name. A bundle's preprocessing.json names the one its model was trained with.

Training and inference must compute the same numbers in the same order. `text5-plus-patch-index-v1` is
the training team's schema (feature/nfe-predictor-training: features/basic.py, training/dataset.py
make_patch_vector). Its text part is copied here line for line, so inference needs no torch or training
code; tests/test_features.py pins it against values produced by the training code.

`basic_text_v0` is the old request-level placeholder, kept for the dummy bundle.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np


class ContextError(ValueError):
    """The context is missing something this extractor needs (reported as invalid_input)."""


@dataclass(frozen=True)
class Extractor:
    name: str
    feature_names: tuple[str, ...]
    fn: Callable[[str, Mapping[str, Any]], Sequence[float]]
    context_keys: tuple[str, ...] = field(default=())  # required context keys besides "text"

    def check_context(self, context: Mapping[str, Any]) -> None:
        """Raise ContextError if a required key is missing or invalid."""
        if "patch_index" in self.context_keys:
            i = context.get("patch_index")
            if isinstance(i, bool) or not isinstance(i, int) or i < 0:
                raise ContextError(f"extractor {self.name} needs a non-negative int 'patch_index', got {i!r}")

    def __call__(self, text: str, context: Mapping[str, Any]) -> np.ndarray:
        x = np.asarray(self.fn(text, context), dtype=np.float64)
        if x.shape != (len(self.feature_names),):
            raise ValueError(f"extractor {self.name} returned shape {x.shape}, "
                             f"expected ({len(self.feature_names)},)")
        return x


# --- text5-plus-patch-index-v1 (training schema) -----------------------------------------------

TEXT5_NAMES = ("character_count", "word_count", "punctuation_count", "digit_count", "average_word_length")


@lru_cache(maxsize=256)
def _text5(text: str) -> tuple[float, ...]:
    """Same as training's features.basic.extract_basic_features, in FEATURE_NAMES order.

    Cached: a request asks once per patch with the same text, so only patch_index changes.
    """
    words = text.split()
    word_count = len(words)
    average_word_length = sum(len(word) for word in words) / word_count if word_count else 0.0
    return (
        float(len(text)),
        float(word_count),
        float(sum(character in string.punctuation for character in text)),
        float(sum(character.isdigit() for character in text)),
        float(average_word_length),
    )


def _text5_plus_patch_index(text: str, context: Mapping[str, Any]) -> list[float]:
    return [*_text5(text), float(context["patch_index"])]


# --- basic_text_v0 (placeholder, request level) ------------------------------------------------

_SENTENCE_END = re.compile(r"[.!?]+")
_OTHER_PUNCT = re.compile(r"[,;:\-()\"']")


def _basic_text_v0(text: str, context: Mapping[str, Any]) -> list[float]:
    """PLACEHOLDER, not the team's extractor: simple counts so the pipeline is testable."""
    digits = sum(c.isdigit() for c in text)
    return [
        float(len(text)),
        float(len(text.split())),
        float(len(_SENTENCE_END.findall(text))),
        float(len(_OTHER_PUNCT.findall(text))),
        float(digits),
        float(digits > 0),
    ]


EXTRACTORS: Dict[str, Extractor] = {
    "text5-plus-patch-index-v1": Extractor(
        "text5-plus-patch-index-v1", TEXT5_NAMES + ("patch_index",), _text5_plus_patch_index,
        context_keys=("patch_index",),
    ),
    "basic_text_v0": Extractor(
        "basic_text_v0",
        ("n_chars", "n_words", "n_sentence_ends", "n_other_punct", "n_digits", "has_number"),
        _basic_text_v0,
    ),
}


def get_extractor(name: str) -> Extractor:
    try:
        return EXTRACTORS[name]
    except KeyError:
        raise KeyError(f"unknown feature extractor {name!r}; known: {sorted(EXTRACTORS)}") from None
