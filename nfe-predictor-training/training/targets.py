"""Strict label ingestion. Do not invent patch labels from utterance WER/SIM."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from training.contracts import CANDIDATE_NFES, SCHEMA_VERSION, validate_budgets, validate_observation


@dataclass(frozen=True)
class PatchTarget:
    comparison_id: str
    prompt_id: str
    prompt_family_id: str
    speaker_id: str
    text: str
    patch_index: int
    targets: tuple[float, ...]   # 0.0 or 1.0 where observed; 0.0 placeholder where unknown
    mask: tuple[int, ...]        # 1 = known pass/fail, 0 = not measured
    all_tested_failed: bool      # true ONLY if ALL candidate budgets were observed and failed
    budgets: tuple[int, ...] = CANDIDATE_NFES
    schema_version: str = SCHEMA_VERSION
    patch_id: str | None = None
    experiment_id: str | None = None
    # Retain scorer values, validity flags and exact generation provenance.
    observations: tuple[dict[str, Any], ...] = ()


def read_observations(path: str | Path, budgets=CANDIDATE_NFES) -> list[dict[str, Any]]:
    """Read one JSON object per (comparison_id, NFE)."""
    data = []
    with Path(path).open(encoding="utf-8") as fp:
        for i, line in enumerate(fp, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                    validate_observation(row, budgets)
                    data.append(row)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Bad JSON at line {i}: {e}") from e
                except ValueError as e:
                    raise ValueError(f"Invalid observation at line {i}: {e}") from e
    return data


def build_targets(
    observations: Iterable[dict[str, Any]],
    budgets: tuple[int, ...] = CANDIDATE_NFES,
) -> list[PatchTarget]:
    """Aggregate NFE outcomes from identical counterfactual patch contexts.

    comparison_id identifies one exact preceding generation state and fixed
    initial noise; the different NFE rows must compare the same context.
    quality_pass is bool for verified labels or null when unmeasured. No data
    from an unlabeled observation can produce a positive/negative target.
    """
    validate_budgets(budgets)
    groups: dict[str, list[dict[str, Any]]] = {}
    prompts: dict[str, tuple[str, str]] = {}
    for row in observations:
        validate_observation(row, budgets)
        prompt = (row["prompt_family_id"], row["text"])
        if row["prompt_id"] in prompts and prompts[row["prompt_id"]] != prompt:
            raise ValueError("A prompt_id must have consistent text and prompt_family_id")
        prompts[row["prompt_id"]] = prompt
        groups.setdefault(row["comparison_id"], []).append(row)

    out = []
    for cid, rows in sorted(groups.items()):
        first = rows[0]
        identity = (
            "schema_version", "experiment_id", "patch_id", "prompt_id",
            "prompt_family_id", "speaker_id", "text", "patch_index", "seed",
            "checkpoint_id", "conditioning_id", "anchor_state_hash", "noise_hash",
            "solver", "feature_schema_version", "pre_generation_features",
        )
        for r in rows:
            if any(r.get(k) != first.get(k) for k in identity):
                raise ValueError(f"Mixed patch contexts in comparison_id={cid}")
        outcomes: dict[int, bool | None] = {}
        label_protocol = None
        for r in rows:
            nfe = r["nfe"]
            if nfe in outcomes:
                raise ValueError(f"Duplicate observation for {cid} / NFE {nfe}")
            if r["quality_pass"] is not None:
                if label_protocol is not None and r["label_protocol"] != label_protocol:
                    raise ValueError(f"Mixed label protocols in comparison_id={cid}")
                label_protocol = r["label_protocol"]
            outcomes[nfe] = r["quality_pass"]
        mask = tuple(int(outcomes.get(b) is not None) for b in budgets)
        target = tuple(float(outcomes.get(b) is True) for b in budgets)
        out.append(PatchTarget(
            comparison_id=cid, prompt_id=first["prompt_id"],
            prompt_family_id=first["prompt_family_id"],
            speaker_id=first["speaker_id"], text=first["text"],
            patch_index=first["patch_index"], targets=target,
            mask=mask, all_tested_failed=(all(mask) and not any(target)),
            budgets=budgets, schema_version=first["schema_version"],
            patch_id=first.get("patch_id"), experiment_id=first.get("experiment_id"),
            observations=tuple(deepcopy(r) for r in sorted(rows, key=lambda r: r["nfe"])),
        ))
    return out
