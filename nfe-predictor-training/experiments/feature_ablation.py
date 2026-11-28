"""Feature-ablation experiments for the NFE predictor.

Question: do richer text representations improve budget selection enough to
justify their extraction cost? For each feature-set spec (experiments.feature_sets),
on the same family-grouped splits and seeds, this reports:

- test masked BCE and Brier score (how well pass probabilities are predicted)
- budget selection: the validation-chosen threshold applied to test, giving mean NFE
  and pass rate against a fixed budget on the same patches (training.evaluate.compare_fixed_nfe)
- cost: input dimensions, extra numbers to ship, and per-text extraction time

From nfe-predictor-training/:

    python -m experiments.feature_ablation --legacy-dry-run            # harness check only
    python -m experiments.feature_ablation --observations reviewed.jsonl --out report.json

A report built from legacy dry-run labels has "conclusive": false; do not quote it as a result.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

from experiments.ablation_data import (
    AblationDataset, AblationExample, SPLITS, from_patch_targets, legacy_utterance_dry_run,
)
from experiments.ablation_train import (
    DEFAULT_THRESHOLDS, SplitArrays, fit_standardizer, oracle_mean_nfe, score_seed, train_mlp,
)
from experiments.sequence_cost import COST_MODELS, CostModel, DEFAULT_POLICIES, policy as check_policy
from experiments.feature_sets import BASE_FITTERS, Fitter, FittedFeatures, build, with_static_embeddings
from features.embeddings import StaticEmbeddings
from training.contracts import FALLBACK_NFE, REFERENCE_NFE
from training.targets import build_targets, read_observations
from training.train import TrainingConfig

DEFAULT_SPECS = ("text5", "rich", "text5+rich", "pca4(text5+rich)")
_ROOT = Path(__file__).resolve().parents[1]
LEGACY_CLIPS = _ROOT.parent / "quality-evals/wer_cer/results/nfe_benchmark/clips_arush.csv"
LEGACY_SPLIT = _ROOT / "configs/benchmark_54_split_v1.csv"
DRY_RUN_WARNING = ("Labels are a provisional dry run on the legacy full-utterance sweep, "
                   "not reviewed patch labels, and each utterance is priced as a single patch. "
                   "Use this to check the harness, not to compare features.")


def _unique_texts(examples: Sequence[AblationExample]) -> list[str]:
    return list(dict.fromkeys(e.text for e in examples))


def _raw_matrix(fitted: FittedFeatures, examples: Sequence[AblationExample]) -> np.ndarray:
    """Text features plus patch_index, the same layout as make_patch_vector."""
    text = fitted.matrix([e.text for e in examples])
    patch = np.array([[float(e.patch_index)] for e in examples])
    return np.hstack([text, patch])


def _split_arrays(fitted: FittedFeatures, dataset: AblationDataset) -> dict[str, SplitArrays]:
    raw = {s: _raw_matrix(fitted, dataset.split(s)) for s in SPLITS}
    mean, std = fit_standardizer(raw["train"])
    return {
        s: SplitArrays((raw[s] - mean) / std,
                       np.array([e.targets for e in dataset.split(s)], dtype=np.float64),
                       np.array([e.mask for e in dataset.split(s)], dtype=np.float64),
                       tuple(e.sequence_id for e in dataset.split(s)),
                       tuple(e.patch_index for e in dataset.split(s)))
        for s in SPLITS
    }


def cost_profile(fitted: FittedFeatures, texts: Sequence[str], repeats: int = 3) -> dict:
    """Per-text extraction wall time after fitting (what serving would pay per request)."""
    times = []
    for _ in range(repeats):
        for text in texts:
            start = time.perf_counter_ns()
            fitted(text)
            times.append((time.perf_counter_ns() - start) / 1e6)
    times.sort()
    return {"dims": len(fitted.names), "shipped_params": fitted.shipped_params,
            "extract_ms_median": statistics.median(times),
            "extract_ms_p95": times[min(len(times) - 1, int(0.95 * len(times)))]}


def _mean_std(values: Sequence[float]) -> dict | None:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0, "n": len(values)}


def _summarize(seed_results: Sequence[dict]) -> dict:
    selected = [r["selection"] for r in seed_results if r["selection"]]
    return {
        "test_loss": _mean_std([r["test_loss"] for r in seed_results]),
        "test_brier": _mean_std([r["test_brier"] for r in seed_results]),
        "adaptive_mean_nfe": _mean_std([s["adaptive_mean_nfe"] for s in selected]),
        "adaptive_pass_rate": _mean_std([s["adaptive_pass_rate"] for s in selected]),
        "fixed_pass_rate": _mean_std([s["fixed_pass_rate"] for s in selected]),
        "paired_coverage": _mean_std([s["paired_coverage"] for s in selected]),
        "seeds_without_threshold": sum(r["selection"] is None for r in seed_results),
        "sequence_selection": _summarize_sequences(seed_results),
    }


def _summarize_sequences(seed_results: Sequence[dict]) -> list[dict]:
    """Per cost model: test speed and quality of the validation-chosen policy, across seeds."""
    by_cost: dict[str, list[dict]] = {}
    for r in seed_results:
        for sel in r.get("sequence_selection", []):
            by_cost.setdefault(sel["cost_model"], []).append(sel)
    out = []
    for name, sels in by_cost.items():
        chosen = [s for s in sels if s["test"] is not None]
        out.append({
            "cost_model": name,
            "speedup_vs_fixed": _mean_std([s["speedup_vs_fixed"] for s in chosen]),
            "request_ms": _mean_std([s["test"]["mean_request_ms"] for s in chosen]),
            "fixed_request_ms": _mean_std([s["fixed_test"]["mean_request_ms"] for s in sels]),
            "switches": _mean_std([s["test"]["mean_switches"] for s in chosen]),
            "pass_rate": _mean_std([s["test"]["pass_rate"] for s in chosen]),
            "fixed_pass_rate": _mean_std([s["test"]["fixed_pass_rate"] for s in chosen]),
            "paired_coverage": _mean_std([s["test"]["paired_coverage"] for s in chosen]),
            "seeds_with_estimated_cost": sum(s["test"]["cost_estimated_outside_measured_nfes"] for s in chosen),
            "policies": dict(Counter(f"{s['policy']}@{s['threshold']}" for s in chosen)),
            "seeds_without_choice": len(sels) - len(chosen),
        })
    return out


def run_feature_set(
    dataset: AblationDataset, spec: str, seeds: Sequence[int], config: TrainingConfig, *,
    fixed_nfe: int, fallback_nfe: int, thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    base_fitters: Mapping[str, Fitter] = BASE_FITTERS,
    cost_models: Sequence[CostModel] = (), policies: Sequence[str] = DEFAULT_POLICIES,
) -> dict:
    fitted = build(spec, base_fitters)(_unique_texts(dataset.split("train")))
    arrays = _split_arrays(fitted, dataset)
    seed_results = []
    for seed in seeds:
        seed_config = replace(config, seed=seed)
        model = train_mlp(arrays["train"], arrays["val"], seed_config)
        result = score_seed(model, arrays["val"], arrays["test"], dataset.budgets,
                            fixed_nfe=fixed_nfe, fallback_nfe=fallback_nfe, thresholds=thresholds,
                            cost_models=cost_models, policies=policies)
        seed_results.append({"seed": seed, **result})
    return {"spec": fitted.spec, **cost_profile(fitted, _unique_texts(dataset.examples)),
            "summary": _summarize(seed_results), "seeds": seed_results}


def run_ablation(
    dataset: AblationDataset, specs: Sequence[str] = DEFAULT_SPECS, seeds: Sequence[int] = (0, 1, 2),
    config: TrainingConfig = TrainingConfig(), *, fixed_nfe: int = REFERENCE_NFE,
    fallback_nfe: int = FALLBACK_NFE, thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    base_fitters: Mapping[str, Fitter] = BASE_FITTERS,
    cost_models: Sequence[CostModel] = (COST_MODELS["nfe"], COST_MODELS["optimized-rtx5060"]),
    policies: Sequence[str] = DEFAULT_POLICIES,
) -> dict:
    if not seeds:
        raise ValueError("need at least one seed")
    for name in policies:
        check_policy(name)
    for nfe in (fixed_nfe, fallback_nfe):
        if nfe not in dataset.budgets:
            raise ValueError(f"NFE {nfe} is not on the dataset budget axis {dataset.budgets}")
    test = _split_arrays(build("text5")([]), dataset)["test"]
    return {
        "label_source": dataset.label_source, "conclusive": dataset.conclusive,
        "warning": None if dataset.conclusive else DRY_RUN_WARNING,
        "budgets": list(dataset.budgets), "fixed_nfe": fixed_nfe, "fallback_nfe": fallback_nfe,
        "examples": {s: len(dataset.split(s)) for s in SPLITS},
        "observed_per_budget": {
            s: dict(zip(map(str, dataset.budgets), map(int, np.sum([e.mask for e in dataset.split(s)], axis=0))))
            for s in SPLITS
        },
        "test_oracle_mean_nfe": oracle_mean_nfe(test, dataset.budgets, fallback_nfe),
        "seeds": list(seeds), "training_config": asdict(config), "thresholds": list(thresholds),
        "feature_sets": [run_feature_set(dataset, s, seeds, config, fixed_nfe=fixed_nfe,
                                         fallback_nfe=fallback_nfe, thresholds=thresholds,
                                         base_fitters=base_fitters, cost_models=cost_models,
                                         policies=policies)
                         for s in specs],
        "cost_models": {c.name: c.source for c in cost_models}, "policies": list(policies),
    }


def _fmt(stat: dict | None, digits: int = 3) -> str:
    return "n/a" if stat is None else f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def to_markdown(report: dict) -> str:
    lines = []
    if report["warning"]:
        lines += [f"> **Not conclusive.** {report['warning']}", ""]
    lines += [
        f"Labels: `{report['label_source']}`. Examples (train/val/test): "
        + "/".join(str(report["examples"][s]) for s in SPLITS)
        + f". Seeds: {len(report['seeds'])}. Fixed and fallback NFE: {report['fixed_nfe']}/{report['fallback_nfe']}. "
        + f"Test oracle mean NFE: {report['test_oracle_mean_nfe']:.1f}.",
        "",
        "| feature set | dims | extract ms p50 / p95 | test Brier | test loss "
        "| adaptive mean NFE | adaptive pass | fixed pass | seeds w/o threshold |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for fs in report["feature_sets"]:
        s = fs["summary"]
        lines.append(
            f"| `{fs['spec']}` | {fs['dims']} | {fs['extract_ms_median']:.3f} / {fs['extract_ms_p95']:.3f} "
            f"| {_fmt(s['test_brier'])} | {_fmt(s['test_loss'])} | {_fmt(s['adaptive_mean_nfe'], 1)} "
            f"| {_fmt(s['adaptive_pass_rate'], 2)} | {_fmt(s['fixed_pass_rate'], 2)} "
            f"| {s['seeds_without_threshold']} |")
    if any(fs["summary"]["sequence_selection"] for fs in report["feature_sets"]):
        lines += [
            "", "Budget selection with switch cost. Policy and threshold are chosen on validation. Pass rates "
            "are on patches with both outcomes measured, from per-patch labels, so they are estimates. "
            "`*`: some chosen NFEs were not measured by the cost model, so their cost is extrapolated.", "",
            "| feature set | cost model | speedup vs fixed | est. ms/request (fixed) "
            "| switches/request | pass (fixed) | chosen policy@threshold (seeds) |",
            "|---|---|---|---|---|---|---|",
        ]
        for fs in report["feature_sets"]:
            for sel in fs["summary"]["sequence_selection"]:
                fixed_ms = sel["fixed_request_ms"]["mean"] if sel["fixed_request_ms"] else float("nan")
                fixed_pass = sel["fixed_pass_rate"]["mean"] if sel["fixed_pass_rate"] else float("nan")
                chosen = ", ".join(f"{k} ({v})" for k, v in sel["policies"].items()) or "none"
                star = "*" if sel["seeds_with_estimated_cost"] else ""
                lines.append(
                    f"| `{fs['spec']}` | {sel['cost_model']} | {_fmt(sel['speedup_vs_fixed'], 2)}{star} "
                    f"| {_fmt(sel['request_ms'], 0)} ({fixed_ms:.0f}) | {_fmt(sel['switches'], 2)} "
                    f"| {_fmt(sel['pass_rate'], 2)} ({fixed_pass:.2f}) | {chosen} |")
        lines += ["", *(f"- `{name}`: {source}" for name, source in report["cost_models"].items())]
    return "\n".join(lines) + "\n"


def _load(args) -> AblationDataset:
    if args.observations:
        return from_patch_targets(build_targets(read_observations(args.observations)), seed=args.split_seed)
    return legacy_utterance_dry_run(args.legacy_clips, args.legacy_split, wer_max=args.wer_max)


def _cost_model(name: str) -> CostModel:
    if name.strip() not in COST_MODELS:
        raise SystemExit(f"unknown cost model {name!r}; known: {sorted(COST_MODELS)}")
    return COST_MODELS[name.strip()]


def _load_embeddings(pairs: Sequence[str]) -> dict[str, StaticEmbeddings]:
    models = {}
    for pair in pairs:
        name, sep, directory = pair.partition("=")
        if not sep or not name.strip() or not directory.strip():
            raise SystemExit(f"--static-embeddings expects NAME=DIR, got {pair!r}")
        if name in models:
            raise SystemExit(f"--static-embeddings name {name!r} given twice")
        models[name] = StaticEmbeddings.load(directory, name=name)
    return models


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--observations", type=Path, help="reviewed patch-nfe-v2 JSONL")
    source.add_argument("--legacy-dry-run", action="store_true", help="legacy sweep, harness check only")
    parser.add_argument("--legacy-clips", type=Path, default=LEGACY_CLIPS)
    parser.add_argument("--legacy-split", type=Path, default=LEGACY_SPLIT)
    parser.add_argument("--wer-max", type=float, default=0.1, help="dry-run pass rule")
    parser.add_argument("--features", default=",".join(DEFAULT_SPECS), help="comma-separated specs")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=TrainingConfig().epochs)
    parser.add_argument("--fixed-nfe", type=int, default=REFERENCE_NFE)
    parser.add_argument("--fallback-nfe", type=int, default=FALLBACK_NFE)
    parser.add_argument("--static-embeddings", action="append", default=[], metavar="NAME=DIR",
                        help="add a model2vec directory as base feature set NAME (repeatable)")
    parser.add_argument("--cost-models", default="nfe,optimized-rtx5060",
                        help=f"comma-separated, from {sorted(COST_MODELS)}")
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES),
                        help="comma-separated: per_patch, request_max, min_runR")
    parser.add_argument("--out", type=Path, help="write the full JSON report here")
    args = parser.parse_args(argv)

    report = run_ablation(
        _load(args), [s for s in args.features.split(",") if s.strip()],
        [int(s) for s in args.seeds.split(",")], TrainingConfig(epochs=args.epochs),
        fixed_nfe=args.fixed_nfe, fallback_nfe=args.fallback_nfe,
        base_fitters=with_static_embeddings(_load_embeddings(args.static_embeddings)),
        cost_models=[_cost_model(n) for n in args.cost_models.split(",") if n.strip()],
        policies=[p.strip() for p in args.policies.split(",") if p.strip()],
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    sys.stdout.write(to_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
