"""A dummy bundle in the real format, for tests and for wiring before a trained model exists.

The dummy model only looks at the word count: short text passes at low NFE, long text needs more,
very long text passes nowhere (-> fallback). It is NOT a trained model.

    python -m nfe_predictor.testing /tmp/predictor-dummy
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Union

import numpy as np

from nfe_predictor.features import get_extractor

DUMMY_BUDGETS = (4, 8, 16, 32)


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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m nfe_predictor.testing OUT_DIR")
    print(make_dummy_bundle(sys.argv[1]))
