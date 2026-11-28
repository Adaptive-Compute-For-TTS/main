"""Inputs for the feature ablation: reviewed patch targets, or a legacy dry run.

Real comparisons use reviewed patch-nfe-v2 targets (`from_patch_targets`).
`legacy_utterance_dry_run` reads the 54-prompt full-utterance sweep only to exercise
the harness before patch labels exist. Its provisional labels are never written out
as training labels, and every report built from them carries `conclusive: false`.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import csv
from dataclasses import dataclass
from pathlib import Path

from training.contracts import (
    CANDIDATE_NFES, EXPERIMENT_SCHEMA_VERSION, SPLIT_RATIOS, SPLIT_SEED, validate_budgets,
)
from training.splits import grouped_prompt_splits, validate_split_assignments
from training.targets import PatchTarget

SPLITS = ("train", "val", "test")
LEGACY_LABEL_SOURCE = "legacy-utterance-dry-run-v1"


@dataclass(frozen=True)
class AblationExample:
    example_id: str
    prompt_id: str
    family_id: str
    text: str
    patch_index: int
    targets: tuple[float, ...]  # 1.0 pass / 0.0 fail where observed
    mask: tuple[int, ...]       # 1 = observed, 0 = unknown
    sequence_id: str = ""       # one generated request; patches are ordered by patch_index

    def __post_init__(self):
        if not self.sequence_id:
            object.__setattr__(self, "sequence_id", self.example_id)


@dataclass(frozen=True)
class AblationDataset:
    examples: tuple[AblationExample, ...]
    assignments: dict[str, str]  # prompt_id -> split, grouped by family
    budgets: tuple[int, ...]
    label_source: str
    conclusive: bool  # False: dry-run labels, do not report as results

    def split(self, name: str) -> tuple[AblationExample, ...]:
        """Labeled examples of one split. Unknown-only examples are dropped."""
        if name not in SPLITS:
            raise ValueError(f"unknown split {name!r}")
        return tuple(e for e in self.examples
                     if self.assignments[e.prompt_id] == name and any(e.mask))


def _check_nonempty_splits(dataset: AblationDataset) -> AblationDataset:
    for name in SPLITS:
        if not dataset.split(name):
            raise ValueError(f"the {name} split has no labeled examples")
    return dataset


_SEQUENCE_KEYS = ("experiment_id", "prompt_id", "speaker_id", "seed",
                  "checkpoint_id", "conditioning_id", "solver")


def _sequence_id(target: PatchTarget) -> str:
    """Patches from the same generated request share these generation settings."""
    first = target.observations[0] if target.observations else {}
    return "|".join(str(first.get(k)) for k in _SEQUENCE_KEYS)


def from_patch_targets(
    targets: Sequence[PatchTarget], *, assignments: dict[str, str] | None = None,
    seed: int = SPLIT_SEED, ratios=SPLIT_RATIOS,
) -> AblationDataset:
    """Wrap reviewed targets from training.targets.build_targets, split by family."""
    if not targets:
        raise ValueError("no patch targets")
    budgets = targets[0].budgets
    if any(t.budgets != budgets for t in targets):
        raise ValueError("all targets must share one budget axis")
    records = [{"prompt_id": t.prompt_id, "prompt_family_id": t.prompt_family_id} for t in targets]
    split_map = dict(assignments) if assignments is not None else grouped_prompt_splits(records, seed, ratios)
    validate_split_assignments(records, split_map)
    protocols = {r["label_protocol"] for t in targets for r in t.observations
                 if r.get("quality_pass") is not None}
    if len(protocols) != 1:
        raise ValueError(f"need exactly one observed label protocol, got {sorted(protocols)}")
    examples = tuple(
        AblationExample(t.comparison_id, t.prompt_id, t.prompt_family_id, t.text,
                        t.patch_index, t.targets, t.mask, _sequence_id(t))
        for t in targets
    )
    conclusive = all(t.schema_version == EXPERIMENT_SCHEMA_VERSION for t in targets)
    return _check_nonempty_splits(
        AblationDataset(examples, split_map, budgets, protocols.pop(), conclusive))


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def _legacy_outcome(row: dict[str, str] | None, cap_s: float, wer_max: float) -> tuple[float, int]:
    """(target, mask) for one clip. Clips cut off by the length cap are unknown:
    the cap, not the NFE, decided them."""
    if row is None:
        return 0.0, 0
    if row["asr_status"] != "ok":
        return 0.0, 1  # silent or unreadable audio is a failure
    if float(row["duration_s"]) >= cap_s - 1e-6 or row["wer_no_intro"] == "":
        return 0.0, 0
    return float(float(row["wer_no_intro"]) <= wer_max), 1


def legacy_utterance_dry_run(
    clips_csv: str | Path, split_csv: str | Path, *, wer_max: float = 0.1,
    budgets: tuple[int, ...] = CANDIDATE_NFES,
) -> AblationDataset:
    """One request-level example per (run, prompt) from quality-evals' clips_arush.csv.

    Provisional rule, for exercising the harness only: pass = readable audio, not cut
    off by the run's length cap, and WER (intro removed) <= wer_max.
    """
    validate_budgets(budgets)
    clips = _read_csv(clips_csv)
    splits = {r["prompt_id"]: r for r in _read_csv(split_csv)}
    caps: dict[str, float] = defaultdict(float)
    for row in clips:
        caps[row["run_id"]] = max(caps[row["run_id"]], float(row["duration_s"]))
    groups: dict[tuple[str, str], dict[int, dict[str, str]]] = defaultdict(dict)
    for row in clips:
        if row["prompt_id"] not in splits:
            raise ValueError(f"prompt {row['prompt_id']} is missing from {split_csv}")
        groups[(row["run_id"], row["prompt_id"])][int(row["nfe"])] = row
    examples = []
    for (run_id, prompt_id), by_nfe in sorted(groups.items()):
        outcomes = [_legacy_outcome(by_nfe.get(b), caps[run_id], wer_max) for b in budgets]
        examples.append(AblationExample(
            f"{run_id}:{prompt_id}", prompt_id, splits[prompt_id]["prompt_family_id"],
            next(iter(by_nfe.values()))["text"], 0,
            tuple(t for t, _ in outcomes), tuple(m for _, m in outcomes),
        ))
    assignments = {pid: r["split"] for pid, r in splits.items()
                   if any(e.prompt_id == pid for e in examples)}
    validate_split_assignments(
        [{"prompt_id": e.prompt_id, "prompt_family_id": e.family_id} for e in examples], assignments)
    return _check_nonempty_splits(AblationDataset(
        tuple(examples), assignments, budgets, f"{LEGACY_LABEL_SOURCE}(wer_max={wer_max})", False))
