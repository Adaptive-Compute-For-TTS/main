"""Patch-level input adapter that leaves Shivani's text extractor untouched."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import Dataset

from training.contracts import BASE_TEXT_FEATURES


def make_patch_vector(
    text: str, patch_index: int,
    extract_features: Callable[[str, Mapping[str, Any] | None], Mapping[str, float]],
    context: Mapping[str, Any] | None = None,
) -> list[float]:
    """Canonical 5 pre-generation features plus zero-based patch index.

    Caller injects features.extractor.extract_features to avoid changing
    Shivani's existing implementation. Future hidden-state features must be
    added through a versioned schema, not silently appended.
    """
    if type(patch_index) is not int or patch_index < 0:
        raise ValueError("patch_index must be a nonnegative integer")
    values = extract_features(text, context)
    if set(values) != set(BASE_TEXT_FEATURES):
        raise ValueError("Feature-name mismatch; check Shivani's FEATURE_NAMES")
    vector = [float(values[name]) for name in BASE_TEXT_FEATURES]
    return vector + [float(patch_index)]


class PatchTrainingDataset(Dataset):
    """Load aggregated PatchTarget objects after real labels exist."""

    def __init__(self, examples: Sequence, extract_features: Callable) -> None:
        self.examples = list(examples)
        self.extract_features = extract_features
        if not self.examples or not all(any(e.mask) for e in self.examples):
            raise ValueError("Unlabeled examples cannot enter supervised training")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        e = self.examples[index]
        x = make_patch_vector(e.text, e.patch_index, self.extract_features)
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(e.targets, dtype=torch.float32),
            torch.tensor(e.mask, dtype=torch.float32),
        )
