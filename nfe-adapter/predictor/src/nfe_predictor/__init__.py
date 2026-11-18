"""Inference package for the NFE budget predictor: text -> the NFE a request runs at.

    predictor = load_predictor("models/predictor-v1", fallback_nfe=32)
    decision = predictor.predict_budget({"text": "..."})   # never raises
    decision.nfe, decision.fallback_reason, decision.as_dict()
"""

__version__ = "0.1.0"

from nfe_predictor.bundle import Bundle, BundleError, load_bundle  # noqa: E402
from nfe_predictor.predictor import (  # noqa: E402
    FALLBACK_REASONS,
    BudgetDecision,
    BudgetPredictor,
    FallbackPredictor,
    load_predictor,
)

__all__ = ["FALLBACK_REASONS", "BudgetDecision", "BudgetPredictor", "Bundle", "BundleError",
           "FallbackPredictor", "__version__", "load_bundle", "load_predictor"]
