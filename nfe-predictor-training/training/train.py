"""Reusable training foundations for the existing experiments/ package.

No experiments run on import. Callers supply verified patch observations,
experiment settings and artifact destinations; no serving policy is generated.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.mlp import PatchNFEPredictor, masked_bce_loss
from training.contracts import (
    CANDIDATE_NFES, EXPERIMENT_SCHEMA_VERSION, MODEL_FEATURES, SPLIT_RATIOS,
    SPLIT_SEED, validate_budgets,
)
from training.dataset import FeatureNormalizer, PatchTrainingDataset
from training.evaluate import evaluate
from training.splits import grouped_prompt_splits, validate_split_assignments
from training.targets import PatchTarget, build_targets


@dataclass(frozen=True)
class TrainingConfig:
    """Reproducible baseline defaults, not tuned experiment recommendations."""

    seed: int = 42
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 0

    def __post_init__(self):
        for name in ("seed", "epochs", "batch_size", "num_workers"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name in ("epochs", "batch_size") else 0):
                raise ValueError(f"Invalid {name}")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")


@dataclass
class PreparedData:
    datasets: dict[str, PatchTrainingDataset]
    normalizer: FeatureNormalizer
    assignments: dict[str, str]
    budgets: tuple[int, ...]
    label_protocol: str
    source_sha256: str
    split_seed: int | None
    split_ratios: tuple[float, float, float] | None
    excluded_unlabeled: dict[str, int]


def prepare_data(
    examples: list[PatchTarget], *, seed=SPLIT_SEED, ratios=SPLIT_RATIOS,
    assignments: dict[str, str] | None = None, budgets=CANDIDATE_NFES,
) -> PreparedData:
    """Split all families first, then filter unknown-only patches and fit train scaling.

    New training uses v2 provenance. Legacy v1 records remain readable via
    build_targets, but require actual collection metadata before this pipeline.
    """
    validate_budgets(budgets)
    if not examples or any(e.schema_version != EXPERIMENT_SCHEMA_VERSION for e in examples):
        raise ValueError("New training requires strict patch-nfe-v2 examples")
    if any(e.budgets != budgets or not e.observations for e in examples):
        raise ValueError("Examples must retain observations on the configured budget axis")
    raw = [r for e in examples for r in e.observations]
    # Revalidate provenance and aggregate consistency, including duplicate IDs.
    canonical = build_targets(raw, budgets)
    if {e.comparison_id: e for e in canonical} != {e.comparison_id: e for e in examples} or len(canonical) != len(examples):
        raise ValueError("Examples do not match their source observations")
    protocols = {r["label_protocol"] for r in raw if r["quality_pass"] is not None}
    if len(protocols) != 1:
        raise ValueError("A training dataset requires exactly one observed labeling protocol version")
    records = [{"prompt_id": e.prompt_id, "prompt_family_id": e.prompt_family_id} for e in examples]
    supplied_assignments = assignments is not None
    if assignments is None:
        assignments = grouped_prompt_splits(records, seed, ratios)
    else:
        assignments = dict(assignments)
    validate_split_assignments(records, assignments)
    partitions = {name: [] for name in ("train", "val", "test")}
    excluded = dict.fromkeys(partitions, 0)
    for e in sorted(examples, key=lambda e: e.comparison_id):
        name = assignments[e.prompt_id]
        if any(e.mask):
            partitions[name].append(e)
        else:
            excluded[name] += 1
    for name, rows in partitions.items():
        if not rows:
            raise ValueError(f"The {name} partition has no observed patch labels")
    normalizer = FeatureNormalizer.fit(partitions["train"])
    datasets = {
        name: PatchTrainingDataset(rows, normalizer=normalizer, budgets=budgets)
        for name, rows in partitions.items()
    }
    source_sha = hashlib.sha256(json.dumps(
        sorted(raw, key=lambda r: (r["comparison_id"], r["nfe"])),
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    return PreparedData(
        datasets, normalizer, assignments, budgets, protocols.pop(), source_sha,
        None if supplied_assignments else seed,
        None if supplied_assignments else ratios, excluded,
    )


def seed_everything(seed: int) -> None:
    """Seed Python/NumPy/Torch and request deterministic kernels on this platform."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _seed_worker(worker_id):
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, *, batch_size=32, shuffle=False, seed=42, num_workers=0):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed), worker_init_fn=_seed_worker,
    )


@dataclass
class TrainingResult:
    model: PatchNFEPredictor
    history: list[dict]
    best_epoch: int


def fit(data: PreparedData, config=TrainingConfig(), *, model=None, device="cpu") -> TrainingResult:
    """Fit masked BCE, select by validation loss, leave the test set untouched.

    An experiment may inject its own initialized baseline model. In that case,
    the experiment must seed before initialization to reproduce starting weights.
    """
    seed_everything(config.seed)
    model = (model if model is not None else PatchNFEPredictor(
        input_dim=len(MODEL_FEATURES), num_budgets=len(data.budgets),
    )).to(device)
    architecture = _model_config(model)
    if architecture["input_dim"] != len(MODEL_FEATURES) or architecture["num_budgets"] != len(data.budgets):
        raise ValueError("Model dimensions do not match the canonical dataset axes")
    train_loader = make_loader(
        data.datasets["train"], batch_size=config.batch_size, shuffle=True,
        seed=config.seed, num_workers=config.num_workers,
    )
    val_loader = make_loader(
        data.datasets["val"], batch_size=config.batch_size, seed=config.seed,
        num_workers=config.num_workers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history, best_state, best_epoch, best_loss = [], None, 0, float("inf")
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss, observed = 0.0, 0
        for x, targets, mask in train_loader:
            x, targets, mask = x.to(device), targets.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_bce_loss(model(x), targets, mask)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            count = int(mask.sum())
            total_loss += float(loss.detach()) * count
            observed += count
        validation = evaluate(model, val_loader, data.budgets)
        history.append({"epoch": epoch, "train_loss": total_loss / observed, "val_loss": validation["loss"]})
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return TrainingResult(model, history, best_epoch)


def _model_config(model):
    if not isinstance(model, PatchNFEPredictor):
        raise ValueError("Expected the existing PatchNFEPredictor baseline")
    layers = [m for m in model.network if isinstance(m, nn.Linear)]
    return {
        "input_dim": layers[0].in_features, "num_budgets": layers[-1].out_features,
        "hidden_dims": [m.out_features for m in layers[:-1]],
        "dropout": next(m.p for m in model.network if isinstance(m, nn.Dropout)),
    }


def save_artifacts(
    result: TrainingResult, data: PreparedData, config: TrainingConfig, directory: str | Path,
    *, model_version: str, experiment_config: dict | None = None,
) -> None:
    """Save evaluation weights and a NumPy export, not a deployable policy bundle.

    W matrices are transposed to Tony's proposed [input, output] NPZ layout.
    Thresholds/fallback require team agreement, so policy.json is never invented.
    This checkpoint is for evaluation; optimizer resume state is not included.
    """
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Use an empty artifact directory to preserve existing experiments")
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("An explicit model_version is required")
    architecture = _model_config(result.model)
    if architecture["input_dim"] != len(MODEL_FEATURES) or architecture["num_budgets"] != len(data.budgets):
        raise ValueError("Artifact model dimensions do not match the dataset")
    metadata = {
        "schema_version": 1, "model_type": "mlp_npz", "model_version": model_version,
        "prediction_unit": "audio_patch", "deployment_ready": False,
        "output_semantics": "independent_pass_probabilities",
        "source_schema_version": EXPERIMENT_SCHEMA_VERSION,
        "feature_names": list(MODEL_FEATURES), "budgets": list(data.budgets),
        "label_protocol": data.label_protocol, "source_sha256": data.source_sha256,
        "split_assignments": data.assignments, "split_seed": data.split_seed,
        "split_strategy": "provided_assignments" if data.split_seed is None else "grouped_by_prompt_family",
        "split_ratios": list(data.split_ratios) if data.split_ratios is not None else None,
        "excluded_unlabeled": data.excluded_unlabeled,
        "observed_per_budget": {
            split: {str(b): sum(e.mask[i] for e in dataset.examples) for i, b in enumerate(data.budgets)}
            for split, dataset in data.datasets.items()
        },
        "training_config": asdict(config), "model_config": architecture,
        "best_epoch": result.best_epoch, "history": result.history,
        "experiment_config": deepcopy(experiment_config or {}),
        "versions": {"torch": str(torch.__version__), "numpy": np.__version__},
    }
    preprocessing = data.normalizer.to_dict()
    # Check JSON serialization before creating any artifacts.
    metadata_json = json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    preprocessing_json = json.dumps(preprocessing, indent=2, sort_keys=True, allow_nan=False) + "\n"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metadata.json").write_text(metadata_json, encoding="utf-8")
    (directory / "preprocessing.json").write_text(preprocessing_json, encoding="utf-8")
    torch.save({
        "model_state_dict": {k: v.detach().cpu() for k, v in result.model.state_dict().items()},
        "metadata": metadata, "preprocessing": preprocessing,
    }, directory / "checkpoint.pt")
    arrays = {}
    for i, layer in enumerate(m for m in result.model.network if isinstance(m, nn.Linear)):
        arrays[f"W{i}"] = layer.weight.detach().cpu().numpy().T.copy()
        arrays[f"b{i}"] = layer.bias.detach().cpu().numpy().copy()
    np.savez(directory / "model.npz", **arrays)


def load_checkpoint(path: str | Path):
    """Restore evaluation weights and their exact preprocessing, on CPU."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    metadata = saved["metadata"]
    validate_budgets(tuple(metadata["budgets"]))
    if (metadata["prediction_unit"] != "audio_patch"
            or tuple(metadata["feature_names"]) != MODEL_FEATURES
            or metadata["model_config"]["input_dim"] != len(MODEL_FEATURES)
            or metadata["model_config"]["num_budgets"] != len(metadata["budgets"])):
        raise ValueError("Checkpoint feature/budget dimensions do not match its schema")
    model = PatchNFEPredictor(**metadata["model_config"])
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return model, FeatureNormalizer.from_dict(saved["preprocessing"]), metadata
