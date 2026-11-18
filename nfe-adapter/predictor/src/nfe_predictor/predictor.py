"""predict_budget: one request's text -> the NFE it runs at, plus why. Never raises."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np

from nfe_predictor.bundle import Bundle, load_bundle

log = logging.getLogger(__name__)

FALLBACK_REASONS = (
    "no_confident_budget",
    "invalid_input",
    "text_too_long",
    "feature_error",
    "model_error",
    "invalid_prediction",
    "predictor_unavailable",
    "internal_error",
)


@dataclass(frozen=True)
class BudgetDecision:
    nfe: int
    fallback_reason: Optional[str]  # None: the model's choice was used
    model_version: Optional[str]
    policy_version: Optional[str]
    bundle_sha256: Optional[str]
    decision_ms: float
    probabilities: Optional[dict]  # {"4": p, "8": p, ...}
    error: Optional[str] = None

    @property
    def used_fallback(self) -> bool:
        return self.fallback_reason is not None

    def as_dict(self) -> dict:
        return asdict(self)


class BudgetPredictor:
    """A loaded bundle. Build once (BudgetPredictor.load / load_predictor), call predict_budget per request."""

    def __init__(self, bundle: Bundle) -> None:
        self.bundle = bundle

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BudgetPredictor":
        """Raises BundleError for a bad bundle. load_predictor() turns that into a fallback predictor."""
        return cls(load_bundle(path))

    @property
    def budgets(self) -> tuple[int, ...]:
        """Every NFE predict_budget can return (the worker warms these up)."""
        return self.bundle.policy.budgets

    @property
    def fallback_nfe(self) -> int:
        return self.bundle.policy.fallback_nfe

    def predict_budget(self, context: Mapping[str, Any]) -> BudgetDecision:
        start = time.perf_counter()
        probabilities: Optional[dict] = None

        def decide(nfe: int, reason: Optional[str], error: Optional[str] = None) -> BudgetDecision:
            b = self.bundle
            return BudgetDecision(nfe=int(nfe), fallback_reason=reason, model_version=b.model_version,
                                  policy_version=b.policy.policy_version, bundle_sha256=b.sha256,
                                  decision_ms=round((time.perf_counter() - start) * 1e3, 4),
                                  probabilities=probabilities, error=error)

        fallback = self.bundle.policy.fallback_nfe
        try:
            text = context.get("text") if isinstance(context, Mapping) else None
            if not isinstance(text, str) or not text.strip():
                return decide(fallback, "invalid_input", "context needs a non-empty string 'text'")
            if len(text) > self.bundle.policy.max_text_chars:
                return decide(fallback, "text_too_long",
                              f"{len(text)} chars > max_text_chars {self.bundle.policy.max_text_chars}")

            try:
                x = self.bundle.features(text)
            except Exception as e:  # noqa: BLE001
                return decide(fallback, "feature_error", f"{type(e).__name__}: {e}")
            if not np.all(np.isfinite(x)):
                return decide(fallback, "feature_error", "non-finite feature values")

            try:
                p = np.asarray(self.bundle.model.predict_proba(x), dtype=np.float64)
            except Exception as e:  # noqa: BLE001
                return decide(fallback, "model_error", f"{type(e).__name__}: {e}")
            budgets = self.bundle.policy.budgets
            if p.shape != (len(budgets),) or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
                return decide(fallback, "invalid_prediction", f"bad model output {p.tolist()!r}")
            probabilities = {str(nfe): round(float(v), 6) for nfe, v in zip(budgets, p)}

            nfe, reason = self.bundle.policy.choose(p)
            return decide(nfe, reason)
        except Exception as e:  # noqa: BLE001  never let the predictor fail a run
            log.exception("predict_budget failed")
            return decide(fallback, "internal_error", f"{type(e).__name__}: {e}")


class FallbackPredictor:
    """Stands in when the bundle can't be loaded: always the fallback NFE, with the load error."""

    def __init__(self, fallback_nfe: int, error: str) -> None:
        self.fallback_nfe = int(fallback_nfe)
        self.error = error

    @property
    def budgets(self) -> tuple[int, ...]:
        return (self.fallback_nfe,)

    def predict_budget(self, context: Mapping[str, Any]) -> BudgetDecision:
        return BudgetDecision(nfe=self.fallback_nfe, fallback_reason="predictor_unavailable", model_version=None,
                              policy_version=None, bundle_sha256=None, decision_ms=0.0, probabilities=None,
                              error=self.error)


def load_predictor(path: Union[str, Path], *, fallback_nfe: int) -> Union[BudgetPredictor, FallbackPredictor]:
    """Load a bundle for serving. If it can't load, log why and return a FallbackPredictor, so the
    service still runs (at fallback_nfe) instead of failing at startup."""
    try:
        return BudgetPredictor.load(path)
    except Exception as e:  # noqa: BLE001
        log.error("predictor bundle %s failed to load, every request falls back to nfe=%d: %s", path, fallback_nfe, e)
        return FallbackPredictor(fallback_nfe, f"{type(e).__name__}: {e}")
