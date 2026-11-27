# nfe-adapter/predictor

Inference package for the shared NFE predictor: given the text (and, for patch-level models, which
audio patch is next), pick the NFE budget to run.

```python
from nfe_predictor import load_predictor

predictor = load_predictor("models/predictor-v1", fallback_nfe=32)   # once, at worker startup
predictor.prediction_unit   # "audio_patch" (training's model) or "request" (old dummy)

# patch level: once per patch, inside generation
decision = predictor.predict_budget({"text": "Call me at 555-0142 on Friday.", "patch_index": 0})  # never raises
decision.nfe               # e.g. 16: this patch runs at this NFE
decision.fallback_reason   # None, or why the safe fallback NFE was used
decision.as_dict()         # JSON-safe, for the runs table and telemetry
```

Owner: Tony. Status: **draft, matches the training spec; waiting on trained weights and policy values**.
- Implements the training team's input schema `text5-plus-patch-index-v1` (Shivani's 5 text features
  + `patch_index`), checked against values from the training code (`tests/test_features.py`).
- Loads training's export (`save_artifacts`) unchanged, and serves a subset of its 9 heads
  (`tests/test_training_export_compat.py`).
- The GPU worker does not call it per patch yet (see open questions); it refuses patch-level bundles.

The GPU worker uses request-level bundles today: `adaptive` jobs call `predict_budget` once and then run
on the existing executor (see `gpu-workers/README.md`).

## Where it sits

```
text ──► features (shared extractor, same code as training)       Shivani, Afreed
     ──► model: pass probability for each NFE budget                 Rababb, Eason
     ──► policy: smallest budget whose probability ≥ its threshold   thresholds from Eric
     ──► BudgetDecision {nfe, model_version, decision_ms, fallback_reason, ...}
     ──► GPU worker runs the request at that NFE (executor, eager or optimized/CUDA graphs)
```

The budget is chosen once per request, so generation runs at one fixed NFE, and the existing executor
and worker run it unchanged.

## What I need from each person

| From | What | Where in the bundle |
|---|---|---|
| Rababb, Shivani | trained model, exported as below | `model.npz`, `metadata.json` |
| Shivani (Afreed) | the shared feature extractor as an importable function, plus the names, order and scaling of its features | `preprocessing.json`, registered in `features.py` |
| Eric | thresholds per budget and the fallback budget | `policy.json` |
| Arush, Atharv | what "pass" means (the label), so output `i` = P(pass at `budgets[i]`) | documented in `metadata.json` |
| Eason | export from the experiment framework in this format (or agree an adapter for trees) | `model_type` |

## `predict_budget(context) -> BudgetDecision`

**Input:** a mapping with `text` (str), plus `patch_index` (int ≥ 0, zero-based: the patch about to be
generated) for patch-level bundles. A missing or bad `patch_index` is `invalid_input`. Other keys are
ignored. Text features are cached per text, so per-patch calls only redo the cheap part (median ~0.01 ms).

**Output:** a `BudgetDecision`. Use `.as_dict()` for JSON.

| field | meaning |
|---|---|
| `nfe` | the budget to run: one of the bundle's budgets, or the fallback NFE |
| `fallback_reason` | `None` when the model's choice was used, else one of the reasons below |
| `model_version` | from `metadata.json` |
| `policy_version` | from `policy.json` (thresholds can change without retraining) |
| `bundle_sha256` | hash of the bundle files: exactly which artifacts made this decision |
| `decision_ms` | wall time of the whole call (features + model + policy) |
| `probabilities` | `{"4": 0.31, "8": 0.77, ...}`: pass probability per budget, `null` if not computed |
| `error` | short message when something failed, else `null` |

`predict_budget` never raises. Every problem becomes the fallback NFE plus a reason, so a predictor
problem can't fail a run.

### Fallback reasons

| reason | when |
|---|---|
| `no_confident_budget` | no budget reached its threshold (includes "every budget fails") |
| `invalid_input` | the context isn't a mapping with a non-empty string `text`, or a patch-level bundle got no valid `patch_index` |
| `text_too_long` | the text is longer than the policy's `max_text_chars` (outside the training range) |
| `feature_error` | the extractor raised, or returned the wrong shape or non-finite values |
| `model_error` | the model raised |
| `invalid_prediction` | wrong shape, NaN/inf, or probabilities outside [0, 1] |
| `predictor_unavailable` | the bundle failed to load at startup; `load_predictor` returns a predictor that always falls back, and logs why |
| `internal_error` | anything else (a bug here) |

## Model bundle (proposal)

One folder per model version. The training side writes it; this package only reads it.

```
predictor-v1/
├── metadata.json       what made it
├── preprocessing.json  how text becomes the model's input
├── policy.json         how probabilities become a decision (Eric's calibration)
└── model.npz           the weights
```

`metadata.json` (extra keys, such as training's split and history, are kept and ignored)
```json
{"schema_version": 1, "model_version": "predictor-v1", "model_type": "mlp_npz",
 "prediction_unit": "audio_patch", "budgets": [4, 6, 8, 10, 12, 16, 20, 24, 32],
 "label_protocol": "...", "source_sha256": "..."}
```
- `prediction_unit`: `"audio_patch"` (one decision per patch, needs `patch_index`) or `"request"` (default).
- `budgets`: the NFE of each model output head, in order. Defaults to the policy's budgets.

`preprocessing.json`: must match training exactly. The loader refuses a bundle whose feature names
(and order) differ from what the named extractor produces.
```json
{"extractor": "text5-plus-patch-index-v1",
 "feature_names": ["character_count", "word_count", "punctuation_count", "digit_count",
                   "average_word_length", "patch_index"],
 "mean": [...], "std": [...]}
```
The model's input is `(features - mean) / std`. Every `std` must be > 0.

`policy.json`
```json
{"policy_version": "calib-v1", "budgets": [4, 8, 16, 32], "thresholds": [0.9, 0.9, 0.9, 0.9],
 "fallback_nfe": 32, "max_text_chars": 1000}
```
- `budgets`: the budgets this policy may pick, ascending, each one a model head (a subset is fine:
  serve only budgets with trained, evaluated heads). The backend accepts 4-32, and on the optimized path the
  worker warms each one up.
- `thresholds`: one per budget, each in (0, 1]. The policy picks the smallest budget whose pass
  probability is at least its threshold.
- `fallback_nfe`: the conservative budget. It must be one of `budgets`.

`model.npz` (`model_type: "mlp_npz"`): a plain MLP saved as numpy arrays `W0, b0, W1, b1, ...`
(`W` is `in × out`), with ReLU between layers and a sigmoid on the output. The input size equals the
number of features, and the output size equals the number of budgets. Output `i` is the probability
that the request passes the quality bar at `budgets[i]`.

Numpy only: inference needs no torch, and the loader refuses pickles. Export from PyTorch with
`np.savez(path, W0=layer.weight.T.numpy(), b0=layer.bias.numpy(), ...)`.

## Open questions

Design questions raised by patch-level training (to decide before changing the code):
- **Per-patch API.** How the executor asks for each patch's NFE, and what the context carries
  (`text`, `patch_index`, anything else).
- **When features are computed.** Text features once per request, with only `patch_index` changing per
  patch?
- **Fallback semantics per patch.** Does one failed patch decision fall back for that patch only, or
  for the rest of the request?
- **Deployment budgets.** All 9 candidates, or a subset like 4/8/16/32? Each budget is another
  warmup/CUDA-graph setup on the optimized path, and a subset needs a model trained on it.
- **Extractor sharing.** Import Shivani's `features.extractor` from the training package, or move it to a
  small shared package that both sides install?
- **Request-level mode.** Keep it as a simpler option, or remove it?
- **What goes in the runs table.** Per-patch decisions could be hundreds per request: log all of them,
  or a summary?

Earlier questions:
- **Target shape.** This draft assumes one pass probability per budget. If the model predicts the
  minimum passing budget instead (softmax over budgets + "none"), there are two options:
  - export per-budget probabilities, where P(pass at b) = sum of the classes ≤ b
  - add a `model_type` for it
- **Tree models (Eason).** They'd need a `joblib` adapter (pickles, so trusted bundles only), or ONNX
  for every model. `models.py` has a single place to add loaders.
- **Monotonicity.** Should the policy force probabilities to be non-decreasing in NFE?
- **Feature extractor.** Copied from training (no torch needed here) and pinned by a parity test. If
  training changes it, the schema name changes and the test values are regenerated.
- **Values.** The budgets, thresholds and fallback NFE come from Eric's calibration.

## Layout and checks

- `src/nfe_predictor/`:
  - `predictor.py`: `load_predictor`, `BudgetPredictor`, `BudgetDecision`
  - `bundle.py`: load and validate a bundle
  - `features.py`: extractor registry
  - `models.py`: model adapters
  - `policy.py`: budget choice
  - `testing.py`: dummy bundle
- `tests/`: dummy-bundle tests, no GPU or real model needed

```bash
pip install -e 'nfe-adapter/predictor[test]'
pytest nfe-adapter/predictor/tests
python -m nfe_predictor.testing /tmp/predictor-dummy                  # write a dummy bundle
python -m nfe_predictor /tmp/predictor-dummy "Call me on Friday."     # one decision, as JSON
```
