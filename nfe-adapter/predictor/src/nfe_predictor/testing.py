"""Dummy bundles in the real format, for tests and for wiring before a trained model exists.

make_dummy_bundle: request level (old placeholder extractor). Looks only at the word count: short text
passes at low NFE, long text needs more, very long text passes nowhere (-> fallback).

make_dummy_patch_bundle: patch level, in the training team's export layout (text5-plus-patch-index-v1,
6 -> hidden -> 9 heads, metadata budgets 4..32) plus a policy serving only 4/8/16/32. Word count as
above, and patch 0 is harder than the rest. The 5 heads the policy doesn't serve (6/10/12/20/24) are set
to always "pass", so reading one by mistake changes the decision and fails a test.

Neither is a trained model.

    python -m nfe_predictor.testing /tmp/predictor-dummy [--patch]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Union

import numpy as np

from nfe_predictor.features import get_extractor

DUMMY_BUDGETS = (4, 8, 16, 32)
TRAINING_BUDGETS = (4, 6, 8, 10, 12, 16, 20, 24, 32)  # training/contracts.py CANDIDATE_NFES


def make_dummy_bundle(path: Union[str, Path], *, thresholds=(0.7, 0.7, 0.7, 0.7), fallback_nfe: int = 32,
                      max_text_chars: int = 2000) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    ex = get_extractor("basic_text_v0")
    n = len(ex.feature_names)
    mean, std = np.zeros(n), np.ones(n)
    words = ex.feature_names.index("n_words")
    mean[words], std[words] = 12.0, 8.0

    # hidden = relu(z_words + 3); logit_b = a_b - 1.5 * hidden
    w0 = np.zeros((n, 1)); w0[words, 0] = 1.0
    b0 = np.array([3.0])
    w1 = np.full((1, len(DUMMY_BUDGETS)), -1.5)
    b1 = np.array([3.0, 5.0, 7.0, 12.0])
    np.savez(path / "model.npz", W0=w0, b0=b0, W1=w1, b1=b1)

    (path / "metadata.json").write_text(json.dumps({
        "schema_version": 1, "model_version": "dummy-v0", "model_type": "mlp_npz",
        "label": "DUMMY: not trained, word count only", "trained_on": None}, indent=2))
    (path / "preprocessing.json").write_text(json.dumps({
        "extractor": ex.name, "feature_names": list(ex.feature_names),
        "mean": mean.tolist(), "std": std.tolist()}, indent=2))
    (path / "policy.json").write_text(json.dumps({
        "policy_version": "dummy-calib-v0", "budgets": list(DUMMY_BUDGETS), "thresholds": list(thresholds),
        "fallback_nfe": fallback_nfe, "max_text_chars": max_text_chars}, indent=2))
    return path


def make_dummy_patch_bundle(path: Union[str, Path], *, budgets=DUMMY_BUDGETS, thresholds=None,
                            fallback_nfe: int = 32, max_text_chars: int = 2000) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    ex = get_extractor("text5-plus-patch-index-v1")
    n = len(ex.feature_names)
    words, patch = ex.feature_names.index("word_count"), ex.feature_names.index("patch_index")
    mean, std = np.zeros(n), np.ones(n)
    mean[words], std[words] = 12.0, 8.0

    # h0 = relu(z_words + 3), h1 = relu(1 - patch_index): 1 on patch 0, 0 after
    # logit_b = a_b - 1.5 * h0 - 3 * h1
    w0 = np.zeros((n, 2)); w0[words, 0] = 1.0; w0[patch, 1] = -1.0
    b0 = np.array([3.0, 1.0])
    a = {4: 3.0, 8: 5.0, 16: 7.0, 32: 12.0}
    unserved = 50.0  # untrained heads: "always pass", must never be read
    w1 = np.array([[-1.5 if b in a else 0.0 for b in TRAINING_BUDGETS],
                   [-3.0 if b in a else 0.0 for b in TRAINING_BUDGETS]])
    b1 = np.array([a.get(b, unserved) for b in TRAINING_BUDGETS])
    np.savez(path / "model.npz", W0=w0, b0=b0, W1=w1, b1=b1)

    # Same keys as training/train.py save_artifacts and FeatureNormalizer.to_dict.
    (path / "metadata.json").write_text(json.dumps({
        "schema_version": 1, "model_type": "mlp_npz", "model_version": "dummy-patch-v0",
        "prediction_unit": "audio_patch", "deployment_ready": False,
        "output_semantics": "independent_pass_probabilities",
        "feature_names": list(ex.feature_names), "budgets": list(TRAINING_BUDGETS),
        "label_protocol": "DUMMY: not trained"}, indent=2))
    (path / "preprocessing.json").write_text(json.dumps({
        "extractor": ex.name, "feature_names": list(ex.feature_names),
        "mean": mean.tolist(), "std": std.tolist(), "fit_split": "train", "fit_comparison_ids": []}, indent=2))
    (path / "policy.json").write_text(json.dumps({
        "policy_version": "dummy-patch-calib-v0", "budgets": list(budgets),
        "thresholds": list(thresholds or [0.7] * len(budgets)),
        "fallback_nfe": fallback_nfe, "max_text_chars": max_text_chars}, indent=2))
    return path


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--patch"]
    if len(args) != 1:
        sys.exit("usage: python -m nfe_predictor.testing OUT_DIR [--patch]")
    print((make_dummy_patch_bundle if "--patch" in sys.argv else make_dummy_bundle)(args[0]))
