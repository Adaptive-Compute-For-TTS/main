"""Compare two runs (e.g. baseline vs. a new kernel): speedup per NFE and audio equality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ttsbench.storage import Run


@dataclass(frozen=True)
class Comparison:
    rows: list[dict[str, Any]]
    warnings: list[str]


def _comparability_warnings(a: Run, b: Run) -> list[str]:
    ma, mb = a.manifest, b.manifest
    warnings = []
    for key, label in (
        ("measurement_protocol_version", "measurement protocol"),
        ("prompts_hash", "prompt set"),
    ):
        if ma.get(key) != mb.get(key):
            warnings.append(f"{label} differs ({ma.get(key)} vs {mb.get(key)})")
    gpus = [{w["environment"].get("torch", {}).get("gpu_name") for w in r.workers()} for r in (a, b)]
    if gpus[0] != gpus[1]:
        warnings.append(f"GPU differs ({sorted(map(str, gpus[0]))} vs {sorted(map(str, gpus[1]))})")
    if ma["config"]["measurement"] != mb["config"]["measurement"]:
        warnings.append("measurement settings (warmup/repeats/seed) differ")
    return warnings


def compare_runs(a: Run, b: Run) -> Comparison:
    """Speedup = a / b, so > 1 means `b` is faster."""
    summary_a = {r["nfe"]: r for r in a.summary() if r["length"] == "all"}
    summary_b = {r["nfe"]: r for r in b.summary() if r["length"] == "all"}
    hashes_a = {(s.prompt_id, s.nfe, s.repeat): s.audio_sha256 for s in a.samples()}
    hashes_b = {(s.prompt_id, s.nfe, s.repeat): s.audio_sha256 for s in b.samples()}

    rows = []
    for nfe in sorted(summary_a.keys() & summary_b.keys()):
        la, lb = summary_a[nfe]["latency_ms"], summary_b[nfe]["latency_ms"]
        shared = [k for k in hashes_a.keys() & hashes_b.keys() if k[1] == nfe]
        identical = sum(hashes_a[k] == hashes_b[k] for k in shared)
        rows.append(
            {
                "nfe": nfe,
                "a_p50_ms": la["p50"],
                "b_p50_ms": lb["p50"],
                "speedup_p50": la["p50"] / lb["p50"] if lb["p50"] else float("nan"),
                "a_p90_ms": la["p90"],
                "b_p90_ms": lb["p90"],
                "audio_identical": f"{identical}/{len(shared)}",
            }
        )
    return Comparison(rows=rows, warnings=_comparability_warnings(a, b))
