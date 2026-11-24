"""Reproducible, leakage-resistant splits by prompt family, never by WAV row."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from training.contracts import SPLIT_RATIOS, SPLIT_SEED


def grouped_prompt_splits(
    records: list[dict], seed: int = SPLIT_SEED,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
) -> dict[str, str]:
    """Return prompt_id -> split; keep every prompt family together.

    Families are shuffled, so ratios refer to family count (prompt count can
    differ if family sizes differ). Stratification must be audited separately.
    """
    if abs(sum(ratios) - 1) > 1e-9 or any(x <= 0 for x in ratios):
        raise ValueError("ratios must be positive and sum to 1")
    families: dict[str, set[str]] = defaultdict(set)
    prompt_family: dict[str, str] = {}
    for r in records:
        pid = r["prompt_id"]
        fid = r["prompt_family_id"]
        if pid in prompt_family and prompt_family[pid] != fid:
            raise ValueError(f"Prompt {pid} has inconsistent family ID")
        prompt_family[pid] = fid
        families[fid].add(pid)
    ids = sorted(families)
    if len(ids) < 3:
        raise ValueError("At least 3 independent families required")
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    n_train, n_val = max(1, n_train), max(1, n_val)
    if n_train + n_val >= n:
        n_train, n_val = n-2, 1
    group_split = {f: ("train" if i < n_train else "val" if i < n_train+n_val else "test")
                   for i, f in enumerate(ids)}
    return {pid: group_split[fid] for pid, fid in sorted(prompt_family.items())}


def initial_54_prompt_manifest(seed: int = SPLIT_SEED) -> list[dict]:
    """A balanced split proposal for the verified p001..p054 benchmark set.

    The source has 6 categories x 3 lengths x 3 prompts. Two per cell go to
    train. One held out per cell, alternately val/test; two held-out cells are
    promoted to train, yielding 38/8/8, category/length aware.

    CAUTION: assumes prompt ID is the family; re-group near duplicates before
    using this for final claims or with a larger expanded dataset.
    """
    cats = ("plain", "numbers", "names_rare_words", "acronyms_abbreviations", "prosody", "phonetic")
    lengths = ("short", "medium", "long")
    rng = random.Random(seed)
    rows = []
    for ci, category in enumerate(cats):
        for li, length in enumerate(lengths):
            ids = [f"p{ci*9 + li*3 + k+1:03d}" for k in range(3)]
            rng.shuffle(ids)
            held = "val" if (ci+li) % 2 == 0 else "test"
            if (ci, li) in ((0, 0), (1, 2)):
                held = "train"
            for j, pid in enumerate(ids):
                rows.append({"prompt_id": pid, "prompt_family_id": pid,
                             "category": category, "length_bucket": length,
                             "split": "train" if j < 2 else held})
    rows.sort(key=lambda r: r["prompt_id"])
    assert {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")} == {"train":38, "val":8, "test":8}
    return rows


def export_split_manifest(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as fp:
        wr = csv.DictWriter(fp, fieldnames=("prompt_id", "prompt_family_id", "category", "length_bucket", "split"))
        wr.writeheader()
        wr.writerows(rows)


def manifest_sha256(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
