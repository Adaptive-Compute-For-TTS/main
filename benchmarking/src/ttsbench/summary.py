"""Aggregate samples into per-NFE (and per-NFE x length bucket) latency summaries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

from ttsbench.schema import Prompt, Sample

PERCENTILES = (50, 90, 95, 99)

# A group is flagged noisy when the median per-prompt coefficient of variation of latency exceeds
# this. CV is computed within (prompt, nfe) cells: pooling prompts of different lengths would
# measure input variety, not measurement noise.
NOISY_CV = 0.05


def describe(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    stats = {"n": int(arr.size), "mean": mean, "std": std, "cv": std / mean if mean else 0.0}
    stats |= {"min": float(arr.min()), "max": float(arr.max())}
    stats |= {f"p{p}": float(np.percentile(arr, p)) for p in PERCENTILES}
    return stats


def _deterministic_fraction(samples: list[Sample]) -> float | None:
    """Share of (prompt, nfe) cells whose repeats with equal seeds produced identical audio."""
    by_cell: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    for s in samples:
        by_cell[(s.prompt_id, s.nfe, s.seed)].add(s.audio_sha256)
    cells = [hashes for hashes in by_cell.values()]
    if not cells:
        return None
    return sum(len(h) == 1 for h in cells) / len(cells)


def _within_prompt_cv(samples: list[Sample]) -> float:
    by_prompt: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        by_prompt[s.prompt_id].append(s.latency_ms)
    cvs = [describe(v)["cv"] for v in by_prompt.values() if len(v) > 1]
    return float(np.median(cvs)) if cvs else 0.0


def _group_row(samples: list[Sample]) -> dict[str, Any]:
    latency = describe(s.latency_ms for s in samples)
    within_cv = _within_prompt_cv(samples)
    device = [s.device_ms for s in samples if s.device_ms is not None]
    stage_names = sorted({name for s in samples for name in s.stages_ms})
    return {
        "n": len(samples),
        "latency_ms": latency,
        "device_ms": describe(device) if device else None,
        "rtf": describe(s.rtf for s in samples),
        "audio_seconds_mean": float(np.mean([s.audio_seconds for s in samples])),
        "stages_ms_p50": {
            name: float(np.median([s.stages_ms[name] for s in samples if name in s.stages_ms])) for name in stage_names
        },
        "deterministic_fraction": _deterministic_fraction(samples),
        "within_prompt_cv": within_cv,
        "noisy": within_cv > NOISY_CV,
    }


def summarize(samples: list[Sample], prompts: list[Prompt]) -> list[dict[str, Any]]:
    lengths = {p.id: p.length for p in prompts}
    groups: dict[tuple[int, str], list[Sample]] = defaultdict(list)
    for s in samples:
        groups[(s.nfe, "all")].append(s)
        groups[(s.nfe, lengths.get(s.prompt_id, "unknown"))].append(s)
    order = {"all": 0, "short": 1, "medium": 2, "long": 3}
    keys = sorted(groups, key=lambda k: (k[0], order.get(k[1], 9)))
    return [{"nfe": nfe, "length": length, **_group_row(groups[(nfe, length)])} for nfe, length in keys]


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    lat, rtf, dev = row["latency_ms"], row["rtf"], row["device_ms"] or {}
    return {
        "nfe": row["nfe"],
        "length": row["length"],
        "n": row["n"],
        "latency_p50_ms": round(lat["p50"], 3),
        "latency_p90_ms": round(lat["p90"], 3),
        "latency_p95_ms": round(lat["p95"], 3),
        "latency_mean_ms": round(lat["mean"], 3),
        "latency_std_ms": round(lat["std"], 3),
        "within_prompt_cv": round(row["within_prompt_cv"], 4),
        "device_p50_ms": round(dev["p50"], 3) if dev else "",
        "rtf_p50": round(rtf["p50"], 4),
        "audio_seconds_mean": round(row["audio_seconds_mean"], 3),
        "deterministic_fraction": row["deterministic_fraction"],
        "noisy": row["noisy"],
    }
