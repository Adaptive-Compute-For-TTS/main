"""Versioned, proposed patch-level NFE dataset and feature contract.

Quality labels MUST be produced by a real counterfactual patch-evaluation
protocol, not copied from whole-utterance fixed-NFE benchmark scores.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from features import FEATURE_NAMES

SCHEMA_VERSION = "patch-nfe-v1"
# Strict schema for future collection; v1 remains readable for compatibility.
EXPERIMENT_SCHEMA_VERSION = "patch-nfe-v2"
FEATURE_SCHEMA_VERSION = "text5-plus-patch-index-v1"
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


def validate_budgets(budgets: tuple[int, ...]) -> None:
    """Validate an ordered experimental budget axis, without approving it."""
    if (
        not budgets
        or any(type(b) is not int or b < 1 for b in budgets)
        or tuple(sorted(set(budgets))) != budgets
    ):
        raise ValueError("budgets must be unique sorted positive integers")


def validate_observation(row: dict[str, Any], budgets=CANDIDATE_NFES) -> None:
    """Validate v1 records or the stricter v2 collection proposal.

    Validation checks metadata and contradictions, not audio or replay fidelity.
    Measurements never automatically become labels. See docs/patch_dataset.md.
    """
    validate_budgets(budgets)
    if not isinstance(row, dict):
        raise ValueError("Each observation must be a JSON object")
    version = row.get("schema_version")
    if version not in (SCHEMA_VERSION, EXPERIMENT_SCHEMA_VERSION):
        raise ValueError("Unexpected schema version")
    for key in ("comparison_id", "prompt_id", "prompt_family_id", "speaker_id", "text"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"Missing {key}")
    if type(row.get("patch_index")) is not int or row["patch_index"] < 0:
        raise ValueError("patch_index must be a nonnegative integer")
    if type(row.get("nfe")) is not int or row["nfe"] not in budgets:
        raise ValueError(f"Unsupported candidate NFE: {row.get('nfe')}")
    if "quality_pass" not in row:
        raise ValueError("quality_pass is required; use null for unmeasured budgets")
    label = row["quality_pass"]
    if label is not None and type(label) is not bool:
        raise ValueError("quality_pass must be true, false, or null")
    controlled = version == EXPERIMENT_SCHEMA_VERSION or label is not None
    if controlled:
        if type(row.get("seed")) is not int:
            raise ValueError("Observed quality labels or v2 comparisons require an integer seed")
        for key in ("anchor_state_hash", "noise_hash", "checkpoint_id", "conditioning_id"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Controlled comparisons require {key}")
    if label is not None:
        if not isinstance(row.get("label_protocol"), str) or not row["label_protocol"].strip():
            raise ValueError("Observed quality labels require label_protocol")
    elif row.get("label_protocol") is not None and (
        not isinstance(row["label_protocol"], str) or not row["label_protocol"].strip()
    ):
        raise ValueError("label_protocol must be a version string or null")

    if version == EXPERIMENT_SCHEMA_VERSION:
        for key in ("experiment_id", "patch_id", "solver"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"v2 comparisons require {key}")
        if row.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError("Unexpected feature_schema_version")
        if row.get("record_status") not in ("planned", "measured"):
            raise ValueError("record_status must be planned or measured")
        for key in ("audio_valid", "not_truncated", "counterfactual_reviewed"):
            if key not in row or (row[key] is not None and type(row[key]) is not bool):
                raise ValueError(f"{key} must be true, false, or null")
            if label is not None and row[key] is None:
                raise ValueError(f"Observed v2 labels require a known {key}")
        if label is not None:
            if row["record_status"] != "measured" or not row["counterfactual_reviewed"]:
                raise ValueError("Observed v2 labels require measured, reviewed patches")
            if label and not (row["audio_valid"] and row["not_truncated"]):
                raise ValueError("A passing patch cannot have invalid or truncated audio")
        if "quality_measurements" not in row:
            raise ValueError("v2 comparisons require quality_measurements (possibly empty)")

    for key in ("quality_measurements", "pre_generation_features"):
        values = row.get(key, {})
        if not isinstance(values, dict):
            raise ValueError(f"{key} must be a mapping")
        for name, value in values.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"{key} names must be nonempty strings")
            if value is None and key == "quality_measurements":
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{key} values must be finite numbers")
    if version == EXPERIMENT_SCHEMA_VERSION and row["record_status"] == "planned":
        flags = ("audio_valid", "not_truncated", "counterfactual_reviewed")
        if label is not None or any(row[k] is not None for k in flags) or any(
            v is not None for v in row["quality_measurements"].values()
        ):
            raise ValueError("Planned records cannot contain quality results")
