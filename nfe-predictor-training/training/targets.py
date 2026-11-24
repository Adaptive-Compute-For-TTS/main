"""Strict label ingestion. Do not invent patch labels from utterance WER/SIM."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from training.contracts import CANDIDATE_NFES, SCHEMA_VERSION


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


def read_observations(path: str | Path) -> list[dict[str, Any]]:
    """Read one JSON object per (comparison_id, NFE)."""
    data = []
    with Path(path).open(encoding="utf-8") as fp:
        for i, line in enumerate(fp, 1):
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Bad JSON at line {i}: {e}") from e
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
    if (
        not budgets
        or any(type(b) is not int or b < 1 for b in budgets)
        or tuple(sorted(set(budgets))) != budgets
    ):
        raise ValueError("budgets must be unique sorted integers")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in observations:
        if not isinstance(row, dict):
            raise ValueError("Each observation must be a JSON object")
        if row.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unexpected schema version")
        cid = row.get("comparison_id")
        if not isinstance(cid, str) or not cid:
            raise ValueError("comparison_id is required")
        if type(row.get("nfe")) is not int or row["nfe"] not in budgets:
            raise ValueError(f"Unsupported candidate NFE: {row.get('nfe')}")
        if "quality_pass" not in row:
            raise ValueError("quality_pass is required; use null for unmeasured budgets")
        if row.get("quality_pass") is not None and type(row["quality_pass"]) is not bool:
            raise ValueError("quality_pass must be true, false, or null")
        for key in ("prompt_id", "prompt_family_id", "speaker_id", "text"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"Missing {key}")
        if type(row.get("patch_index")) is not int or row["patch_index"] < 0:
            raise ValueError("patch_index must be a nonnegative integer")
        groups.setdefault(cid, []).append(row)

    out = []
    for cid, rows in sorted(groups.items()):
        first = rows[0]
        identity = ("prompt_id", "prompt_family_id", "speaker_id", "text", "patch_index", "seed", "checkpoint_id", "conditioning_id", "anchor_state_hash", "noise_hash")
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
                if type(r.get("seed")) is not int:
                    raise ValueError("Observed quality labels require an integer seed")
                for required in ("label_protocol", "anchor_state_hash", "noise_hash", "checkpoint_id", "conditioning_id"):
                    if not isinstance(r.get(required), str) or not r[required].strip():
                        raise ValueError(f"Observed quality labels require {required}")
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
        ))
    return out
