"""Feature extractors, by name. A bundle's preprocessing.json names the one its model was trained with.

Training and inference must run the same code. The real extractor is Shivani's shared one: register
it here by importing it, not by copying it. `basic_text_v0` is a placeholder so the package can be
built and tested before that lands.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class Extractor:
    name: str
    feature_names: tuple[str, ...]
    fn: Callable[[str], Sequence[float]]

    def __call__(self, text: str) -> np.ndarray:
        x = np.asarray(self.fn(text), dtype=np.float64)
        if x.shape != (len(self.feature_names),):
            raise ValueError(f"extractor {self.name} returned shape {x.shape}, "
                             f"expected ({len(self.feature_names)},)")
        return x


_SENTENCE_END = re.compile(r"[.!?]+")
_OTHER_PUNCT = re.compile(r"[,;:\-()\"']")


def _basic_text_v0(text: str) -> list[float]:
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
