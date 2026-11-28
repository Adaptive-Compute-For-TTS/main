"""Feature-ablation harness on synthetic fixtures only; no experimental results."""
import csv
import json

import numpy as np
import pytest
import torch

from experiments.ablation_data import (
    LEGACY_LABEL_SOURCE, from_patch_targets, legacy_utterance_dry_run,
)
from experiments.ablation_train import SplitArrays, choose_threshold, oracle_mean_nfe
from experiments.feature_ablation import main, run_ablation, to_markdown
from experiments.feature_sets import RICH_NAMES, build, rich_features
from features import FEATURE_NAMES, extract_features
from training.train import TrainingConfig

TEXTS = ["The kettle is on.", "Call NASA at 555-0142, Friday!", "Dr. Okonkwo met Siobhan... twice.",
         "Is it 5% or $5?", "Unbelievably extraordinary circumstances arose.", "Plain words here."]


# --- feature sets ---------------------------------------------------------------------------

def test_text5_matches_shared_extractor():
    fitted = build("text5")(TEXTS)
    assert fitted.names == FEATURE_NAMES
    for text in TEXTS:
        expected = [extract_features(text)[n] for n in FEATURE_NAMES]
        assert fitted(text).tolist() == expected


def test_rich_features_counts():
    values = dict(zip(RICH_NAMES, rich_features("Call NASA at 555-0142, Friday! Ask Bob—now... ok?")))
    assert values["sentence_count"] == 3
    assert values["comma_count"] == 1
    assert values["question_exclaim_count"] == 2
    assert values["ellipsis_dash_count"] == 2
    assert values["number_token_count"] == 1
    assert values["acronym_token_count"] == 1
    assert values["capitalized_inner_token_count"] == 3  # NASA, Friday, Bob—now
    assert rich_features("") == (0.0,) * len(RICH_NAMES)
    with pytest.raises(TypeError):
        rich_features(None)


@pytest.mark.parametrize("spec, dims", [
    ("text5", 5), ("rich", 10), ("text5+rich", 15), ("pca3(text5+rich)", 3),
    ("text5+pca2(rich)", 7), ("pca2(text5)+pca2(rich)", 4),
])
def test_spec_dimensions(spec, dims):
    fitted = build(spec)(TEXTS)
    assert len(fitted.names) == dims
    assert fitted.matrix(TEXTS).shape == (len(TEXTS), dims)


@pytest.mark.parametrize("spec", ["nope", "pca2(nope)", "text5+(rich", "text5)", "text5+text5"])
def test_bad_specs_raise(spec):
    with pytest.raises(ValueError):
        build(spec)(TEXTS)


def test_pca_is_fitted_on_training_texts_only():
    train = TEXTS[:4]
    fitted = build("pca2(text5+rich)")(train)
    assert np.allclose(fitted.matrix(train).mean(axis=0), 0, atol=1e-9)
    assert fitted.shipped_params == 2 * 15 + 15 + 15
    with pytest.raises(ValueError, match="pca5"):
        build("pca5(text5)")(train)  # 4 texts < 5 components


def test_pca_is_deterministic():
    a = build("pca3(text5+rich)")(TEXTS).matrix(TEXTS)
    b = build("pca3(text5+rich)")(list(TEXTS)).matrix(TEXTS)
    assert np.array_equal(a, b)


# --- data -----------------------------------------------------------------------------------

def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def legacy_files(tmp_path):
    splits = [{"prompt_id": f"p{i}", "prompt_family_id": f"p{i}", "category": "plain",
               "length_bucket": "short", "split": s}
              for i, s in enumerate(["train", "train", "val", "test"])]
    clips = []
    for i in range(4):
        for nfe in (4, 8, 32):
            clips.append({"run_id": "r", "prompt_id": f"p{i}", "nfe": nfe, "text": f"Prompt {i}.",
                          "asr_status": "ok", "duration_s": "3.0", "wer_no_intro": "0.0"})
    clips[0] = {**clips[0], "wer_no_intro": "0.5"}                            # p0 nfe4 fails
    clips[1] = {**clips[1], "duration_s": "9.12"}                             # p0 nfe8 hit the cap
    clips[3] = {**clips[3], "asr_status": "rejected: silent", "wer_no_intro": ""}  # p1 nfe4 silent
    clips[11] = {**clips[11], "duration_s": "9.12"}                           # sets the run cap
    return _write_csv(tmp_path / "clips.csv", clips), _write_csv(tmp_path / "split.csv", splits)


def test_legacy_dry_run_labels(legacy_files):
    data = legacy_utterance_dry_run(*legacy_files, budgets=(4, 8, 32))
    assert not data.conclusive and data.label_source.startswith(LEGACY_LABEL_SOURCE)
    by_prompt = {e.prompt_id: e for e in data.examples}
    assert by_prompt["p0"].targets == (0.0, 0.0, 1.0) and by_prompt["p0"].mask == (1, 0, 1)
    assert by_prompt["p1"].targets == (0.0, 1.0, 1.0) and by_prompt["p1"].mask == (1, 1, 1)
    assert by_prompt["p3"].mask == (1, 1, 0)
    assert [len(data.split(s)) for s in ("train", "val", "test")] == [2, 1, 1]


def test_patch_targets_are_conclusive_and_grouped(synthetic_examples):
    data = from_patch_targets(synthetic_examples)
    assert data.conclusive and data.label_source == "synthetic-unit-test-v1"
    family_split = {}
    for e in data.examples:
        assert family_split.setdefault(e.family_id, data.assignments[e.prompt_id]) == data.assignments[e.prompt_id]


# --- training and selection -----------------------------------------------------------------

def test_choose_threshold_prefers_cheapest_matching_quality():
    budgets = (4, 8)
    y = np.array([[1, 1], [0, 1]], dtype=float)
    val = SplitArrays(np.zeros((2, 1)), y, np.ones_like(y))
    p = torch.tensor([[0.95, 0.99], [0.7, 0.99]])
    # 0.6 sends the second example to NFE 4 (a failure); 0.8 keeps quality at fewer NFE.
    assert choose_threshold(p, val, budgets, fixed_nfe=8, fallback_nfe=8, thresholds=(0.6, 0.8)) == 0.8
    never = SplitArrays(np.zeros((2, 1)), np.array([[0, 1], [0, 1]], float), np.ones((2, 2)))
    assert choose_threshold(torch.tensor([[0.9, 0.1], [0.9, 0.1]]), never, budgets,
                            fixed_nfe=8, fallback_nfe=8, thresholds=(0.5,)) is None


def test_oracle_mean_nfe():
    arrays = SplitArrays(np.zeros((2, 1)), np.array([[0, 1], [0, 0]], float), np.array([[1, 1], [1, 0]], float))
    assert oracle_mean_nfe(arrays, (4, 8), fallback_nfe=8) == 8.0


def _small_run(data):
    return run_ablation(data, ["text5", "pca2(text5+rich)"], seeds=(0, 1),
                        config=TrainingConfig(epochs=2), fixed_nfe=8, fallback_nfe=8)


def test_run_ablation_end_to_end_is_reproducible(synthetic_examples):
    data = from_patch_targets(synthetic_examples)
    first, second = _small_run(data), _small_run(data)
    assert first["conclusive"] and first["warning"] is None
    assert [fs["spec"] for fs in first["feature_sets"]] == ["text5", "pca2(text5+rich)"]
    for a, b in zip(first["feature_sets"], second["feature_sets"]):
        assert a["seeds"] == b["seeds"]  # everything except timing
        assert a["dims"] in (5, 2) and len(a["seeds"]) == 2
    assert "| `text5` | 5 |" in to_markdown(first)
    json.dumps(first, allow_nan=False)


def test_run_ablation_rejects_off_axis_budgets(synthetic_examples):
    with pytest.raises(ValueError, match="budget axis"):
        run_ablation(from_patch_targets(synthetic_examples), ["text5"], fixed_nfe=7)


def test_cli_dry_run_writes_inconclusive_report(legacy_files, tmp_path, capsys, monkeypatch):
    # The fixture's budgets are 4/8/32; the default axis has nine, unobserved ones are masked.
    out = tmp_path / "report.json"
    assert main(["--legacy-dry-run", "--legacy-clips", str(legacy_files[0]),
                 "--legacy-split", str(legacy_files[1]), "--features", "text5,rich",
                 "--seeds", "0", "--epochs", "1", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["conclusive"] is False and report["warning"]
    assert "Not conclusive" in capsys.readouterr().out
