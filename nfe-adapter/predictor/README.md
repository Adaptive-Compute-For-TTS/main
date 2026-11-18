# nfe-adapter/predictor

Inference package for the shared NFE predictor: given a request's text, pick the NFE budget the whole
request runs at, before generation starts.

```python
from nfe_predictor import load_predictor

predictor = load_predictor("models/predictor-v1", fallback_nfe=32)   # once, at worker startup
decision = predictor.predict_budget({"text": "Call me at 555-0142 on Friday."})   # never raises
decision.nfe               # e.g. 16: the request runs at this NFE
decision.fallback_reason   # None, or why the safe fallback NFE was used
decision.as_dict()         # JSON-safe, for the runs table and telemetry
```

Owner: Tony. Status: **draft**.
- The package runs end to end on a dummy model.
- The feature extractor is a placeholder until Shivani's shared one lands.
- The bundle format below is a proposal: please comment before exporting.

The GPU worker already uses this: `adaptive` jobs call `predict_budget` and then run on the existing
executor (see `gpu-workers/README.md`).

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

**Input:** a mapping with `text` (str). Other keys are ignored for now; there's room for speaker,
mode, and so on later.

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
| `invalid_input` | the context isn't a mapping with a non-empty string `text` |
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

`metadata.json`
```json
{"schema_version": 1, "model_version": "predictor-v1", "model_type": "mlp_npz",
 "label": "pass = WER/CER + speaker similarity criteria v1 (Arush, Atharv)",
 "trained_on": "dataset manifest id + git sha"}
```

`preprocessing.json`: must match training exactly. The loader refuses a bundle whose feature names
(and order) differ from what the named extractor produces.
```json
{"extractor": "basic_text_v0", "feature_names": ["n_chars", "n_words", "..."],
 "mean": [...], "std": [...]}
```
The model's input is `(features - mean) / std`. Every `std` must be > 0.

`policy.json`
```json
{"policy_version": "calib-v1", "budgets": [4, 8, 16, 32], "thresholds": [0.9, 0.9, 0.9, 0.9],
 "fallback_nfe": 32, "max_text_chars": 1000}
```
- `budgets`: ascending, each a supported NFE. The backend accepts 4-32, and on the optimized path the
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

- **Target shape.** This draft assumes one pass probability per budget. If the model predicts the
  minimum passing budget instead (softmax over budgets + "none"), there are two options:
  - export per-budget probabilities, where P(pass at b) = sum of the classes ≤ b
  - add a `model_type` for it
- **Tree models (Eason).** They'd need a `joblib` adapter (pickles, so trusted bundles only), or ONNX
  for every model. `models.py` has a single place to add loaders.
- **Monotonicity.** Should the policy force probabilities to be non-decreasing in NFE?
- **Feature extractor.** Replace the `basic_text_v0` placeholder with Shivani's extractor, imported
  rather than copied. Add a parity test: fixed texts must give the same features as training.
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
