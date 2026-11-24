"""Patch-level input adapter that leaves Shivani's text extractor untouched."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch.utils.data import Dataset

from features import extract_features as shared_extract_features
from training.contracts import (
    BASE_TEXT_FEATURES, CANDIDATE_NFES, FEATURE_SCHEMA_VERSION, MODEL_FEATURES,
    validate_budgets,
)
from training.targets import PatchTarget


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
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("Pre-generation features must be finite")
    return vector + [float(patch_index)]


@dataclass(frozen=True)
class FeatureNormalizer:
    """Population standardization fitted on the training partition only."""

    mean: tuple[float, ...]
    std: tuple[float, ...]
    fit_comparison_ids: tuple[str, ...]

    def __post_init__(self):
        if len(self.mean) != len(MODEL_FEATURES) or len(self.std) != len(MODEL_FEATURES):
            raise ValueError("Normalizer must have one mean/std per canonical feature")
        if not all(math.isfinite(x) for x in self.mean + self.std) or any(x <= 0 for x in self.std):
            raise ValueError("Normalizer means must be finite and stds finite and positive")

    @classmethod
    def fit(cls, train_examples: Sequence[PatchTarget], extractor=shared_extract_features):
        if not train_examples:
            raise ValueError("Cannot fit preprocessing without training examples")
        raw = torch.tensor([
            make_patch_vector(e.text, e.patch_index, extractor) for e in train_examples
        ], dtype=torch.float64)
        mean = raw.mean(dim=0)
        std = raw.std(dim=0, correction=0)
        std = torch.where(std == 0, torch.ones_like(std), std)
        return cls(tuple(mean.tolist()), tuple(std.tolist()), tuple(e.comparison_id for e in train_examples))

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim < 1 or features.shape[-1] != len(MODEL_FEATURES):
            raise ValueError("Expected canonical six-input feature vectors")
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("Features must be finite floating-point values")
        mean = features.new_tensor(self.mean)
        std = features.new_tensor(self.std)
        return (features - mean) / std

    def to_dict(self) -> dict:
        return {
            "extractor": FEATURE_SCHEMA_VERSION,
            "feature_names": list(MODEL_FEATURES),
            "mean": list(self.mean), "std": list(self.std),
            "fit_split": "train", "fit_comparison_ids": list(self.fit_comparison_ids),
        }

    @classmethod
    def from_dict(cls, data: Mapping):
        if (data.get("extractor") != FEATURE_SCHEMA_VERSION
                or tuple(data.get("feature_names", ())) != MODEL_FEATURES
                or data.get("fit_split") != "train"):
            raise ValueError("Incompatible preprocessing schema or fit split")
        return cls(tuple(data["mean"]), tuple(data["std"]), tuple(data["fit_comparison_ids"]))


class PatchTrainingDataset(Dataset):
    """Load aggregated PatchTarget objects after real labels exist."""

    def __init__(
        self, examples: Sequence[PatchTarget], extract_features: Callable = shared_extract_features,
        normalizer: FeatureNormalizer | None = None, budgets=CANDIDATE_NFES,
    ) -> None:
        validate_budgets(budgets)
        self.examples = list(examples)
        self.extract_features = extract_features
        self.normalizer = normalizer
        self.budgets = budgets
        if not self.examples or not all(any(e.mask) for e in self.examples):
            raise ValueError("Unlabeled examples cannot enter supervised training")
        for e in self.examples:
            if e.budgets != budgets or len(e.targets) != len(budgets) or len(e.mask) != len(budgets):
                raise ValueError("Example target/mask budget axis does not match the dataset")
            if any(m not in (0, 1) for m in e.mask) or any(
                not math.isfinite(t) or not 0 <= t <= 1 for t in e.targets
            ):
                raise ValueError("Invalid targets or observation mask")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        e = self.examples[index]
        x = make_patch_vector(e.text, e.patch_index, self.extract_features)
        x = torch.tensor(x, dtype=torch.float32)
        if self.normalizer is not None:
            x = self.normalizer.transform(x)
        return (
            x,
            torch.tensor(e.targets, dtype=torch.float32),
            torch.tensor(e.mask, dtype=torch.float32),
        )
