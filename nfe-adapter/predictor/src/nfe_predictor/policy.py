"""Decision policy: per-budget pass probabilities -> one NFE (or the fallback)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class Policy:
    policy_version: str
    budgets: tuple[int, ...]  # ascending
    thresholds: tuple[float, ...]  # one per budget, in (0, 1]
    fallback_nfe: int  # one of budgets
    max_text_chars: int

    def __post_init__(self) -> None:
        b = self.budgets
        if not b or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in b):
            raise ValueError(f"budgets must be positive ints, got {b!r}")
        if list(b) != sorted(set(b)):
            raise ValueError(f"budgets must be strictly ascending, got {b!r}")
        if len(self.thresholds) != len(b) or not all(0 < t <= 1 for t in self.thresholds):
            raise ValueError(f"need one threshold in (0, 1] per budget, got {self.thresholds!r}")
        if self.fallback_nfe not in b:
            raise ValueError(f"fallback_nfe {self.fallback_nfe} is not one of the budgets {b!r}")
        if self.max_text_chars < 1:
            raise ValueError("max_text_chars must be positive")

    def choose(self, probabilities: Sequence[float]) -> tuple[int, Optional[str]]:
        """The smallest budget whose pass probability reaches its threshold; else the fallback."""
        for nfe, p, t in zip(self.budgets, probabilities, self.thresholds):
            if p >= t:
                return nfe, None
        return self.fallback_nfe, "no_confident_budget"
