"""Model adapters: exported weights -> one pass probability per budget.

Every adapter has n_inputs, n_outputs and predict_proba(x) -> 1-D array of n_outputs probabilities,
where x is the standardized feature vector. Add new model types (sklearn trees via joblib, ONNX, ...)
to MODEL_LOADERS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

import numpy as np


class MlpNpz:
    """Plain MLP from model.npz: arrays W0, b0, W1, b1, ... (W is in x out); ReLU between layers,
    sigmoid on the output."""

    def __init__(self, weights: list[np.ndarray], biases: list[np.ndarray]) -> None:
        if not weights or len(weights) != len(biases):
            raise ValueError("model.npz needs matching W0..Wn and b0..bn")
        for i, (w, b) in enumerate(zip(weights, biases)):
            if w.ndim != 2 or b.shape != (w.shape[1],):
                raise ValueError(f"layer {i}: W{i} {w.shape} and b{i} {b.shape} don't fit")
            if i and w.shape[0] != weights[i - 1].shape[1]:
                raise ValueError(f"layer {i}: input size {w.shape[0]} != previous output size "
                                 f"{weights[i - 1].shape[1]}")
        self.weights, self.biases = weights, biases

    @classmethod
    def load(cls, path: Path) -> "MlpNpz":
        with np.load(path, allow_pickle=False) as z:
            n = sum(1 for k in z.files if k.startswith("W"))
            try:
                ws = [z[f"W{i}"].astype(np.float64) for i in range(n)]
                bs = [z[f"b{i}"].astype(np.float64) for i in range(n)]
            except KeyError as e:
                raise ValueError(f"model.npz is missing {e}") from None
        return cls(ws, bs)

    @property
    def n_inputs(self) -> int:
        return self.weights[0].shape[0]

    @property
    def n_outputs(self) -> int:
        return self.weights[-1].shape[1]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        h = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if i < len(self.weights) - 1:
                h = np.maximum(h, 0.0)
        with np.errstate(over="ignore"):  # exp overflow -> probability 0, which is correct
            return 1.0 / (1.0 + np.exp(-h))


# model_type -> (weights file name in the bundle, loader)
MODEL_LOADERS: Dict[str, tuple[str, Callable[[Path], object]]] = {
    "mlp_npz": ("model.npz", MlpNpz.load),
}
