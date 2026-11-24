"""Versioned, proposed patch-level NFE dataset and feature contract.

Quality labels MUST be produced by a real counterfactual patch-evaluation
protocol, not copied from whole-utterance fixed-NFE benchmark scores.
"""
from __future__ import annotations

from dataclasses import dataclass

from features import FEATURE_NAMES

SCHEMA_VERSION = "patch-nfe-v1"
# Candidate experimental budgets from the team's existing 54-prompt sweep.
# The deployment subset is TBD with the executor team.
CANDIDATE_NFES = (4, 6, 8, 10, 12, 16, 20, 24, 32)
BASE_TEXT_FEATURES = FEATURE_NAMES
MODEL_FEATURES = BASE_TEXT_FEATURES + ("patch_index",)
SPLIT_SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)
REFERENCE_NFE = 32  # high-budget evaluation anchor, not guaranteed acceptable
FALLBACK_NFE = 32   # provisional; confirm with Eric and Tony


@dataclass(frozen=True)
class LabelContract:
    """A proposal for review, NOT a team-approved quality criterion."""

    require_audio_valid: bool = True
    require_not_truncated: bool = True
    require_counterfactual_patch_review: bool = True
    # Future scorer-specific WER/CER/speaker-sim thresholds require validation.
    policy_version: str = "UNAPPROVED"
