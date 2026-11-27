"""Load a model bundle and check that its parts fit together. See README.md for the format.

A bundle that loads is internally consistent:
  - every file is present
  - the extractor is known, and its feature names and order match preprocessing.json
  - the model's input size = number of features
  - the model's output heads = metadata "budgets" (or the policy's budgets if metadata has none)
  - the policy is valid and only uses budgets the model has a head for
Anything else raises BundleError at startup, not at request time.

The policy may serve a subset of the model's budgets: training keeps 9 heads but the pilot only labels
4/8/16/32, so the untrained heads are never read.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Union

import numpy as np

from nfe_predictor.features import Extractor, get_extractor
from nfe_predictor.models import MODEL_LOADERS
from nfe_predictor.policy import Policy

SCHEMA_VERSIONS = {1}


class BundleError(ValueError):
    """The bundle is missing a file or its parts don't fit together."""


@dataclass(frozen=True)
class Bundle:
    path: Path
    model_version: str
    model_type: str
    metadata: dict
    extractor: Extractor
    mean: np.ndarray
    std: np.ndarray
    model: Any
    policy: Policy
    sha256: str
    model_budgets: tuple[int, ...]  # one per model output head, in order
    heads: tuple[int, ...]  # model output index for each policy budget

    @property
    def prediction_unit(self) -> str:
        """"audio_patch" (one decision per patch, context needs patch_index) or "request"."""
        return str(self.metadata.get("prediction_unit", "request"))

    def features(self, text: str, context: Mapping[str, Any]) -> np.ndarray:
        """Raw features -> standardized model input, exactly as in training."""
        return (self.extractor(text, context) - self.mean) / self.std

    def policy_probabilities(self, model_output: np.ndarray) -> np.ndarray:
        """The model's outputs for the policy's budgets only, in policy order."""
        return model_output[list(self.heads)]


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise BundleError(f"missing {path.name} in {path.parent}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise BundleError(f"{path.name} is not valid JSON: {e}") from None
    if not isinstance(data, dict):
        raise BundleError(f"{path.name} must be a JSON object")
    return data


def _need(d: dict, key: str, file: str) -> Any:
    if key not in d:
        raise BundleError(f"{file} is missing {key!r}")
    return d[key]


def bundle_sha256(path: Path, files: list[str]) -> str:
    """One hash over the bundle's files (names + bytes), so a decision names its exact artifacts."""
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(name.encode() + b"\0" + (path / name).read_bytes() + b"\0")
    return h.hexdigest()


def load_bundle(path: Union[str, Path]) -> Bundle:
    path = Path(path)
    if not path.is_dir():
        raise BundleError(f"bundle folder not found: {path}")

    meta = _read_json(path / "metadata.json")
    if _need(meta, "schema_version", "metadata.json") not in SCHEMA_VERSIONS:
        raise BundleError(f"unsupported schema_version {meta['schema_version']!r}; supported: {sorted(SCHEMA_VERSIONS)}")
    model_version = str(_need(meta, "model_version", "metadata.json"))
    model_type = _need(meta, "model_type", "metadata.json")
    if model_type not in MODEL_LOADERS:
        raise BundleError(f"unknown model_type {model_type!r}; known: {sorted(MODEL_LOADERS)}")

    pre = _read_json(path / "preprocessing.json")
    try:
        extractor = get_extractor(_need(pre, "extractor", "preprocessing.json"))
    except KeyError as e:
        raise BundleError(str(e.args[0])) from None
    names = tuple(_need(pre, "feature_names", "preprocessing.json"))
    if names != extractor.feature_names:
        raise BundleError(f"preprocessing.json feature_names {list(names)} don't match extractor "
                          f"{extractor.name} {list(extractor.feature_names)} (training/inference mismatch)")
    mean = np.asarray(_need(pre, "mean", "preprocessing.json"), dtype=np.float64)
    std = np.asarray(_need(pre, "std", "preprocessing.json"), dtype=np.float64)
    if mean.shape != (len(names),) or std.shape != (len(names),):
        raise BundleError(f"mean/std need {len(names)} values, got {mean.shape} / {std.shape}")
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(std)) and np.all(std > 0)):
        raise BundleError("mean must be finite and std finite and > 0")

    pol = _read_json(path / "policy.json")
    try:
        policy = Policy(
            policy_version=str(_need(pol, "policy_version", "policy.json")),
            budgets=tuple(_need(pol, "budgets", "policy.json")),
            thresholds=tuple(float(t) for t in _need(pol, "thresholds", "policy.json")),
            fallback_nfe=_need(pol, "fallback_nfe", "policy.json"),
            max_text_chars=int(pol.get("max_text_chars", 1000)),
        )
    except (TypeError, ValueError) as e:
        if isinstance(e, BundleError):
            raise
        raise BundleError(f"policy.json: {e}") from None

    model_file, loader = MODEL_LOADERS[model_type]
    if not (path / model_file).is_file():
        raise BundleError(f"missing {model_file} in {path} (model_type {model_type})")
    try:
        model = loader(path / model_file)
    except Exception as e:  # noqa: BLE001  any loader failure is a bad bundle
        raise BundleError(f"could not load {model_file}: {type(e).__name__}: {e}") from None
    if model.n_inputs != len(names):
        raise BundleError(f"model takes {model.n_inputs} inputs but there are {len(names)} features")
    model_budgets = tuple(meta.get("budgets", policy.budgets))
    if any(isinstance(b, bool) or not isinstance(b, int) for b in model_budgets) or len(set(model_budgets)) != len(model_budgets):
        raise BundleError(f"metadata.json budgets must be distinct ints, got {list(model_budgets)}")
    if model.n_outputs != len(model_budgets):
        raise BundleError(f"model gives {model.n_outputs} outputs but there are {len(model_budgets)} budgets"
                          + (" in metadata.json" if "budgets" in meta else ""))
    unknown = [b for b in policy.budgets if b not in model_budgets]
    if unknown:
        raise BundleError(f"policy.json budgets {unknown} have no model output; model budgets: {list(model_budgets)}")
    heads = tuple(model_budgets.index(b) for b in policy.budgets)

    files = ["metadata.json", "preprocessing.json", "policy.json", model_file]
    return Bundle(path=path, model_version=model_version, model_type=model_type, metadata=meta,
                  extractor=extractor, mean=mean, std=std, model=model, policy=policy,
                  sha256=bundle_sha256(path, files), model_budgets=model_budgets, heads=heads)
