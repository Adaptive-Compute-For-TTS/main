"""CPU training/evaluation checks on synthetic fixtures, not reported results."""
from dataclasses import replace
import json
import random

import numpy as np
import pytest
import torch

from models.mlp import PatchNFEPredictor, masked_bce_loss
from training.contracts import CANDIDATE_NFES, MODEL_FEATURES
from training.evaluate import compare_fixed_nfe, evaluate, fixed_nfe_baseline, per_budget_metrics
from training.train import (
    TrainingConfig, fit, load_checkpoint, make_loader, prepare_data, save_artifacts,
    seed_everything,
)


def test_seed_and_shuffled_loader_reproduce(synthetic_examples):
    seed_everything(17)
    first = (random.random(), np.random.rand(), torch.rand(3))
    seed_everything(17)
    second = (random.random(), np.random.rand(), torch.rand(3))
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])
    data = prepare_data(synthetic_examples)
    one = list(make_loader(data.datasets["train"], shuffle=True, seed=17, batch_size=7))
    two = list(make_loader(data.datasets["train"], shuffle=True, seed=17, batch_size=7))
    assert all(torch.equal(a, b) for batch1, batch2 in zip(one, two) for a, b in zip(batch1, batch2))


def test_fit_reproduces_selects_best_validation_and_never_reads_test(synthetic_examples):
    data = prepare_data(synthetic_examples)

    class ForbiddenTestDataset:
        def __len__(self):
            raise AssertionError("fit accessed the test split")

        def __getitem__(self, index):
            raise AssertionError("fit accessed the test split")

    data.datasets["test"] = ForbiddenTestDataset()
    config = TrainingConfig(seed=7, epochs=3, batch_size=16, learning_rate=0.01)
    one, two = fit(data, config), fit(data, config)
    assert one.history == two.history
    assert all(torch.equal(w, two.model.state_dict()[k]) for k, w in one.model.state_dict().items())
    assert 1 <= one.best_epoch <= config.epochs
    validation = evaluate(one.model, make_loader(data.datasets["val"]), data.budgets)
    assert validation["loss"] == pytest.approx(min(r["val_loss"] for r in one.history))
    assert all(np.isfinite(r["train_loss"]) and np.isfinite(r["val_loss"]) for r in one.history)
    assert one.history[-1]["train_loss"] < one.history[0]["train_loss"]
    assert not one.model.training


def test_fit_rejects_model_feature_dimension_mismatch(synthetic_examples):
    data = prepare_data(synthetic_examples)
    with pytest.raises(ValueError, match="dimensions"):
        fit(data, TrainingConfig(epochs=1), model=PatchNFEPredictor(input_dim=5))


def test_validation_loss_weights_observations_not_batches():
    model = PatchNFEPredictor(dropout=0)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.network[-1].bias.copy_(torch.arange(9, dtype=torch.float32))
    x = torch.zeros(3, 6)
    targets = torch.zeros(3, 9)
    mask = torch.zeros(3, 9)
    mask[0, 0] = mask[1, 0] = 1
    mask[2, :] = 1
    loader = [(x[:2], targets[:2], mask[:2]), (x[2:], targets[2:], mask[2:])]
    expected = float(masked_bce_loss(model(x), targets, mask).detach())
    model.train()
    result = evaluate(model, loader, fixed_nfe=32)
    assert model.training
    assert result["loss"] == pytest.approx(expected)
    assert result["observed"] == 11
    assert result["fixed_nfe_baseline"]["coverage"] == pytest.approx(1 / 3)
    model.eval()
    evaluate(model, loader)
    assert not model.training


def test_per_budget_metrics_never_count_missing_as_failures():
    budgets = (4, 8, 32)
    probabilities = torch.tensor([[0.9, 0.8, 0.9], [0.8, 0.1, 0.2], [0.2, 0.7, 0.1]])
    targets = torch.tensor([[1., 0., 0.], [0., 0., 0.], [0., 1., 0.]])
    mask = torch.tensor([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    metrics = per_budget_metrics(probabilities, targets, mask, budgets)
    assert metrics["4"]["observed"] == 2
    assert metrics["4"]["tp"] == metrics["4"]["fp"] == 1
    assert metrics["4"]["accuracy"] == 0.5
    assert metrics["4"]["brier"] == pytest.approx((0.1**2 + 0.8**2) / 2)
    assert metrics["8"]["observed"] == 1 and metrics["8"]["accuracy"] == 1
    assert metrics["32"]["observed"] == 0
    assert all(metrics["32"][key] is None for key in ("accuracy", "precision", "recall", "brier"))
    baseline = fixed_nfe_baseline(targets, mask, 32, budgets)
    assert baseline["coverage"] == 0 and baseline["pass_rate"] is None


def test_fixed_comparison_uses_same_measured_patch_subset():
    budgets = (4, 8, 32)
    p = torch.tensor([[0.9, 0.1, 0.2], [0.1, 0.9, 0.2], [0.1, 0.1, 0.1]])
    y = torch.tensor([[0., 0., 1.], [0., 0., 1.], [0., 0., 1.]])
    mask = torch.tensor([[1., 0., 1.], [0., 0., 1.], [0., 0., 1.]])
    report = compare_fixed_nfe(p, y, mask, fixed_nfe=32, threshold=0.8, fallback_nfe=32, budgets=budgets)
    assert report["paired_observed"] == 2  # second patch's chosen NFE 8 is unknown
    assert report["paired_coverage"] == pytest.approx(2 / 3)
    assert report["fixed_pass_rate"] == 1
    assert report["adaptive_pass_rate"] == 0.5
    assert report["adaptive_mean_nfe"] == 18
    assert report["fixed_mean_nfe"] == 32


def test_checkpoint_preprocessing_and_npz_round_trip(tmp_path, synthetic_examples):
    data = prepare_data(synthetic_examples)
    config = TrainingConfig(epochs=2, batch_size=16)
    result = fit(data, config)
    destination = tmp_path / "unit-test-only-artifacts"
    save_artifacts(result, data, config, destination, model_version="synthetic-test-only")
    restored, scaler, metadata = load_checkpoint(destination / "checkpoint.pt")
    assert scaler == data.normalizer
    assert metadata["prediction_unit"] == "audio_patch" and metadata["deployment_ready"] is False
    assert metadata["feature_names"] == list(MODEL_FEATURES)
    assert metadata["budgets"] == list(CANDIDATE_NFES)
    assert metadata["split_assignments"] == data.assignments
    assert metadata["observed_per_budget"]["train"]["4"] == len(data.datasets["train"])
    assert metadata["observed_per_budget"]["train"]["32"] == 0
    assert json.loads((destination / "preprocessing.json").read_text()) == scaler.to_dict()
    assert not (destination / "policy.json").exists()
    x, _, _ = next(iter(make_loader(data.datasets["test"], batch_size=5)))
    assert torch.equal(restored(x), result.model(x))
    # Exact adapter math from Tony's documented W[in,out]/ReLU/sigmoid layout.
    with np.load(destination / "model.npz", allow_pickle=False) as arrays:
        h = x.numpy().astype(np.float64)
        for i in range(3):
            h = h @ arrays[f"W{i}"] + arrays[f"b{i}"]
            if i < 2:
                h = np.maximum(h, 0)
        probabilities = 1 / (1 + np.exp(-h))
    np.testing.assert_allclose(probabilities, result.model.probabilities(x).numpy(), rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="empty artifact directory"):
        save_artifacts(result, data, config, destination, model_version="do-not-overwrite")
    mismatched = replace(result, model=PatchNFEPredictor(num_budgets=8))
    with pytest.raises(ValueError, match="dimensions"):
        save_artifacts(mismatched, data, config, tmp_path / "invalid", model_version="invalid")
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("changes", [
    {"epochs": 0}, {"batch_size": 0}, {"num_workers": -1}, {"seed": True},
    {"learning_rate": float("nan")}, {"weight_decay": -0.1},
])
def test_invalid_training_configuration(changes):
    with pytest.raises(ValueError):
        replace(TrainingConfig(), **changes)
