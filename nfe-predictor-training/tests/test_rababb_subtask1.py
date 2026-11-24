import csv
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from features import FEATURE_NAMES, extract_features, features_to_vector
from models.mlp import PatchNFEPredictor, masked_bce_loss, choose_budget
from training.contracts import BASE_TEXT_FEATURES, CANDIDATE_NFES, MODEL_FEATURES
from training.dataset import PatchTrainingDataset, make_patch_vector
from training.targets import build_targets
from training.splits import grouped_prompt_splits, initial_54_prompt_manifest


def test_mlp_shapes_and_loss():
    model = PatchNFEPredictor(len(MODEL_FEATURES), len(CANDIDATE_NFES))
    logits = model(torch.randn(4, len(MODEL_FEATURES)))
    assert logits.shape == (4, 9)
    labels = torch.zeros_like(logits)
    mask = torch.zeros_like(logits)
    mask[:, 1] = 1
    loss = masked_bce_loss(logits, labels, mask)
    assert loss.isfinite()
    loss.backward()
    assert model.network[0].weight.grad is not None
    with pytest.raises(ValueError):
        masked_bce_loss(logits, labels, torch.zeros_like(logits))


def test_selector_fallback():
    assert choose_budget(torch.tensor([.2, .9, .1]), (4, 8, 32), .8, 32) == 8
    assert choose_budget(torch.tensor([.2, .3, .1]), (4, 8, 32), .8, 32) == 32


def observation(nfe, quality_pass, cid="cx"):
    return dict(schema_version="patch-nfe-v1", comparison_id=cid,
                prompt_id="p001", prompt_family_id="p001", speaker_id="spk",
                text="Hello", patch_index=0, checkpoint_id="ckpt",
                conditioning_id="cond", seed=42, anchor_state_hash="hash_state", noise_hash="hash_noise", nfe=nfe, quality_pass=quality_pass,
                label_protocol="test_protocol" if quality_pass is not None else None)


def test_unknown_is_not_failure_and_all_failed_requires_complete_set():
    ex = build_targets([observation(4, False), observation(6, None)])[0]
    assert ex.mask[:3] == (1, 0, 0)
    assert ex.targets[0] == 0.0
    assert not ex.all_tested_failed
    all_failed = build_targets([observation(n, False) for n in CANDIDATE_NFES])[0]
    assert all_failed.all_tested_failed
    assert all_failed.mask == (1,) * 9


def test_duplicate_and_mixed_context_rejected():
    with pytest.raises(ValueError):
        build_targets([observation(4, True), observation(4, False)])
    with pytest.raises(ValueError):
        build_targets([dict(observation(4, True), prompt_id="p002"), observation(8, True)])


def test_shivani_features_compatible_without_edit():
    def extractor(text, context):
        assert text == "Hello!"
        return dict(character_count=6, word_count=1, punctuation_count=1,
                    digit_count=0, average_word_length=6)
    assert make_patch_vector("Hello!", 4, extractor) == [6, 1, 1, 0, 6, 4]


def test_verified_54_manifest_exact_and_reproducible():
    rows = initial_54_prompt_manifest()
    assert len(rows) == 54
    assert rows == initial_54_prompt_manifest()
    assert {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")} == {"train": 38, "val": 8, "test": 8}
    assert len({r["prompt_id"] for r in rows}) == 54
    assert all(any(r["split"] == "train" for r in rows if r["category"] == c and r["length_bucket"] == l) for c in {x["category"] for x in rows} for l in {x["length_bucket"] for x in rows})


def test_group_family_never_leaks():
    rows = [dict(prompt_id=f"p{i}", prompt_family_id=f"g{i//2}") for i in range(12)]
    splits = grouped_prompt_splits(rows)
    assert all(splits[f"p{i}"] == splits[f"p{i+1}"] for i in range(0,12,2))


def test_different_noise_not_same_comparison():
    with pytest.raises(ValueError):
        build_targets([observation(4, True), dict(observation(8, True), noise_hash="different")])

def test_observed_quality_requires_provenance():
    with pytest.raises(ValueError):
        build_targets([dict(observation(4, True), label_protocol=None)])


@pytest.mark.parametrize("text", ["Hi, 2 you!", "", "   \n\t "])
def test_actual_extractor_preserves_canonical_order(text):
    assert BASE_TEXT_FEATURES == FEATURE_NAMES
    assert MODEL_FEATURES == FEATURE_NAMES + ("patch_index",)
    assert make_patch_vector(text, 3, extract_features) == (
        features_to_vector(extract_features(text)) + [3.0]
    )


@pytest.mark.parametrize("patch_index", [True, -1, 1.5])
def test_patch_vector_rejects_invalid_patch_index(patch_index):
    with pytest.raises(ValueError, match="patch_index"):
        make_patch_vector("Hello", patch_index, extract_features)


def test_dataset_batch_with_actual_extractor():
    # Synthetic unit-test observations only; these are not training labels.
    examples = build_targets([observation(4, False), observation(8, True)])
    dataset = PatchTrainingDataset(examples, extract_features)
    features, targets, mask = next(iter(DataLoader(dataset, batch_size=1)))
    assert features.dtype == targets.dtype == mask.dtype == torch.float32
    assert features.tolist() == [features_to_vector(extract_features("Hello")) + [0.0]]
    assert targets.tolist() == [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    assert mask.tolist() == [[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    model = PatchNFEPredictor()
    loss = masked_bce_loss(model(features), targets, mask)
    assert loss.isfinite()
    loss.backward()


def test_dataset_rejects_unlabeled_examples():
    examples = build_targets([observation(4, None)])
    with pytest.raises(ValueError, match="Unlabeled"):
        PatchTrainingDataset(examples, extract_features)


def test_masked_loss_ignores_unknown_budgets():
    logits = torch.tensor([[0.0, 10.0, 0.0]], requires_grad=True)
    targets = torch.tensor([[0.0, 0.0, 1.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    loss = masked_bce_loss(logits, targets, mask)
    assert loss.item() == pytest.approx(0.69314718)
    loss.backward()
    assert logits.grad.tolist() == [[0.25, 0.0, -0.25]]


@pytest.mark.parametrize("seed", [None, True, "42"])
def test_observed_quality_requires_integer_seed(seed):
    with pytest.raises(ValueError, match="integer seed"):
        build_targets([dict(observation(4, True), seed=seed)])


def test_different_label_protocols_cannot_share_targets():
    with pytest.raises(ValueError, match="Mixed label protocols"):
        build_targets([
            observation(4, True),
            dict(observation(8, True), label_protocol="different_protocol"),
        ])


def test_missing_quality_field_has_explicit_error():
    row = observation(4, None)
    del row["quality_pass"]
    with pytest.raises(ValueError, match="quality_pass is required"):
        build_targets([row])


def test_nfe_must_be_integer():
    with pytest.raises(ValueError, match="Unsupported candidate NFE"):
        build_targets([observation(4.0, True)])


def test_manifest_matches_checked_in_benchmark():
    training_root = Path(__file__).resolve().parents[1]
    with (training_root / "configs/benchmark_54_split_v1.csv").open(
        encoding="utf-8", newline=""
    ) as fp:
        rows = list(csv.DictReader(fp))
    assert rows == initial_54_prompt_manifest()
    source = training_root.parent / "quality-evals/wer_cer/prompts/benchmark_prompts_v1.json"
    prompts = {p["id"]: p for p in json.loads(source.read_text(encoding="utf-8"))}
    assert set(prompts) == {r["prompt_id"] for r in rows}
    for row in rows:
        prompt = prompts[row["prompt_id"]]
        assert row["category"] == prompt["category"]
        assert row["length_bucket"] == prompt["length_bucket"]


def test_split_is_independent_of_row_order_and_repeated_measurements():
    rows = [dict(prompt_id=f"p{i}", prompt_family_id=f"g{i//2}") for i in range(12)]
    assert grouped_prompt_splits(rows) == grouped_prompt_splits(list(reversed(rows)) * 3)
    with pytest.raises(ValueError, match="inconsistent family ID"):
        grouped_prompt_splits(rows + [dict(prompt_id="p0", prompt_family_id="other")])
