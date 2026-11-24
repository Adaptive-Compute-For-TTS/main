"""Collection-schema and leakage tests with clearly synthetic observations."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from features import extract_features
from training.contracts import MODEL_FEATURES, validate_observation
from training.dataset import FeatureNormalizer, PatchTrainingDataset, make_patch_vector
from training.splits import grouped_prompt_splits, validate_split_assignments
from training.targets import build_targets, read_observations
from training.train import prepare_data


def test_quality_metadata_retained_without_deriving_labels(synthetic_observation):
    row = synthetic_observation(quality_pass=None, quality_measurements={"test_score": 0.9})
    target = build_targets([row])[0]
    assert not any(target.mask)
    assert target.patch_id == "patch-test"
    assert target.experiment_id == "unit-test-only"
    assert target.observations[0]["quality_measurements"] == {"test_score": 0.9}
    row["quality_measurements"]["test_score"] = 0.1
    assert target.observations[0]["quality_measurements"]["test_score"] == 0.9


@pytest.mark.parametrize("key,value", [
    ("patch_id", "another-patch"), ("experiment_id", "another-experiment"),
    ("patch_index", 1), ("speaker_id", "another-speaker"), ("seed", 43),
    ("checkpoint_id", "another-checkpoint"), ("conditioning_id", "another-config"),
    ("anchor_state_hash", "another-state"), ("noise_hash", "another-noise"),
    ("solver", "another-solver"), ("pre_generation_features", {"planned_patch_count": 3}),
])
def test_comparisons_require_identical_starting_conditions(synthetic_observation, key, value):
    with pytest.raises(ValueError, match="Mixed patch contexts"):
        build_targets([synthetic_observation(4), synthetic_observation(8, **{key: value})])


@pytest.mark.parametrize("key,value", [
    ("seed", None), ("checkpoint_id", ""), ("conditioning_id", None),
    ("anchor_state_hash", None), ("noise_hash", None), ("patch_id", ""),
    ("experiment_id", ""), ("solver", ""), ("audio_valid", "true"),
    ("not_truncated", 1), ("quality_measurements", {"score": float("nan")}),
    ("pre_generation_features", {"extra": float("inf")}),
    ("feature_schema_version", "unknown-schema"),
])
def test_v2_rejects_invalid_metadata_even_without_labels(synthetic_observation, key, value):
    with pytest.raises(ValueError):
        validate_observation(synthetic_observation(quality_pass=None, **{key: value}))


@pytest.mark.parametrize("changes", [
    {"audio_valid": None}, {"audio_valid": False}, {"not_truncated": False},
    {"counterfactual_reviewed": False}, {"label_protocol": None}, {"record_status": "planned"},
])
def test_passing_labels_require_consistent_known_validity(synthetic_observation, changes):
    with pytest.raises(ValueError):
        validate_observation(synthetic_observation(**changes))


def test_invalid_audio_does_not_automatically_become_failure(synthetic_observation):
    unknown = synthetic_observation(quality_pass=None, audio_valid=False, counterfactual_reviewed=False)
    target = build_targets([unknown])[0]
    assert target.targets[0] == 0 and target.mask[0] == 0
    reviewed_failure = synthetic_observation(quality_pass=False, audio_valid=False)
    assert build_targets([reviewed_failure])[0].mask[0] == 1


def test_example_manifest_has_no_quality_results():
    path = Path(__file__).resolve().parents[1] / "configs/patch_experiment_example.jsonl"
    rows = read_observations(path)
    assert len(rows) >= 2
    assert all(r["record_status"] == "planned" and r["quality_pass"] is None for r in rows)
    examples = build_targets(rows)
    assert len(examples) == 1 and not any(examples[0].mask)
    with pytest.raises(ValueError, match="labeling protocol"):
        prepare_data(examples)


def test_reader_reports_bad_record_line(tmp_path, synthetic_observation):
    path = tmp_path / "observations.jsonl"
    path.write_text(json.dumps(synthetic_observation()) + "\n[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_observations(path)


def test_every_patch_speaker_seed_and_budget_stays_in_family_split(synthetic_examples):
    rows = [row for e in synthetic_examples for row in e.observations]
    splits = grouped_prompt_splits(rows)
    assert splits == grouped_prompt_splits(list(reversed(rows)))
    for family in {r["prompt_family_id"] for r in rows}:
        assert len({splits[r["prompt_id"]] for r in rows if r["prompt_family_id"] == family}) == 1
    validate_split_assignments(rows, splits)
    family_rows = [r for r in rows if r["prompt_family_id"] == "family-0"]
    splits[family_rows[0]["prompt_id"]] = "test"
    splits[family_rows[-1]["prompt_id"]] = "train"
    with pytest.raises(ValueError, match="leaks"):
        validate_split_assignments(rows, splits)


def test_unlabeled_family_cannot_hide_split_leakage(synthetic_examples):
    data = prepare_data(synthetic_examples)
    splits = dict(data.assignments)
    splits["p0-0"], splits["p0-1"] = "train", "test"
    rows = [deepcopy(row) for e in synthetic_examples for row in e.observations]
    for row in rows:
        if row["prompt_family_id"] == "family-0":
            row["quality_pass"] = row["label_protocol"] = None
    with pytest.raises(ValueError, match="leaks"):
        prepare_data(build_targets(rows), assignments=splits)


def test_normalizer_fits_train_only_and_handles_constant_features(synthetic_examples):
    data = prepare_data(synthetic_examples)
    train_examples = data.datasets["train"].examples
    raw = torch.tensor([
        make_patch_vector(e.text, e.patch_index, extract_features) for e in train_examples
    ], dtype=torch.float64)
    assert data.normalizer.mean == tuple(raw.mean(0).tolist())
    assert set(data.normalizer.fit_comparison_ids) == {e.comparison_id for e in train_examples}
    assert all(s > 0 for s in data.normalizer.std)
    assert torch.isfinite(data.normalizer.transform(raw)).all()
    altered = []
    for e in synthetic_examples:
        rows = deepcopy(e.observations)
        if data.assignments[e.prompt_id] != "train":
            for row in rows:
                row["text"] = "Held-out extreme text! " * 1000
        altered.extend(rows)
    other = prepare_data(build_targets(altered), assignments=data.assignments)
    assert data.normalizer == other.normalizer
    restored = FeatureNormalizer.from_dict(data.normalizer.to_dict())
    assert restored == data.normalizer
    bad = restored.to_dict()
    bad["feature_names"] = list(reversed(MODEL_FEATURES))
    with pytest.raises(ValueError, match="preprocessing"):
        FeatureNormalizer.from_dict(bad)


def test_budget_order_and_aggregates_cannot_silently_change(synthetic_examples):
    with pytest.raises(ValueError, match="budget axis"):
        PatchTrainingDataset([replace(synthetic_examples[0], budgets=(4, 8))])
    e = synthetic_examples[0]
    modified = replace(e, targets=(0.0,) * 9)
    with pytest.raises(ValueError, match="source observations"):
        prepare_data([modified] + synthetic_examples[1:])


def test_mixed_protocols_rejected_across_training_examples(synthetic_examples):
    rows = [deepcopy(row) for e in synthetic_examples for row in e.observations]
    rows[0]["label_protocol"] = "different-review-protocol"
    rows[1]["label_protocol"] = "different-review-protocol"
    with pytest.raises(ValueError, match="exactly one"):
        prepare_data(build_targets(rows))


def test_unknown_only_patches_excluded_after_split(synthetic_examples):
    data = prepare_data(synthetic_examples)
    e = synthetic_examples[0]
    rows = deepcopy(e.observations)
    for row in rows:
        row["comparison_id"] = "unknown-only-test"
        row["quality_pass"] = row["label_protocol"] = None
    other = prepare_data(synthetic_examples + build_targets(rows), assignments=data.assignments)
    assert sum(other.excluded_unlabeled.values()) == 1
    assert other.normalizer == data.normalizer
    assert other.split_seed is None and other.split_ratios is None


def test_no_silent_fallback_when_a_split_has_no_labels(synthetic_examples):
    data = prepare_data(synthetic_examples)
    rows = [deepcopy(row) for e in synthetic_examples for row in e.observations]
    for row in rows:
        if data.assignments[row["prompt_id"]] == "test":
            row["quality_pass"] = row["label_protocol"] = None
    with pytest.raises(ValueError, match="test partition has no observed"):
        prepare_data(build_targets(rows), assignments=data.assignments)


@pytest.mark.parametrize("ratios", [(0.5, 0.5), (0.7, 0.2, 0.2), (float("nan"), 0.5, 0.5)])
def test_invalid_split_ratios_fail_explicitly(ratios):
    with pytest.raises(ValueError, match="ratios"):
        grouped_prompt_splits([], ratios=ratios)
