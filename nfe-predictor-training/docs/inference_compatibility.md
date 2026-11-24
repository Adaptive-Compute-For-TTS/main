# Patch baseline and request-level inference

Reviewed `origin/nfe-predictor-inference` at commit
`daeddc314150c2ca141463ccb8d4a2780d12bfc4` using Git, without modifying that branch.
Tony's `nfe-adapter/predictor` is a draft that chooses one budget per request.

| Contract | Training baseline | Current inference package |
| --- | --- | --- |
| Prediction unit | One controlled audio patch | Entire request, chosen once before generation |
| Inputs | Shivani's canonical five text features, then `patch_index` | Text-only `basic_text_v0` placeholder with six different features |
| Outputs | Independent pass probabilities for each experimental budget | Independent probabilities, then one request-wide NFE choice |
| Normalization | Train-only mean/std on the six canonical patch inputs | Mean/std from `preprocessing.json`, validated against its registered extractor |
| Hidden states | Absent | Absent; additional context keys currently ignored |

The input widths both happen to be six, but their meanings and ordering differ.
Passing `patch_index` to the current `predict_budget(context)` does not help: it
uses `bundle.features(text)`, whose extractor takes only text. Do not register a
fake text-only extractor that fills the missing patch index with zero.

The export preserves the practical parts of Tony's proposed layout:

- `model.npz`: `W0,b0,W1,b1,W2,b2`, with each W shaped `[input, output]`;
  ReLU between layers and independent sigmoid after the last layer. Training
  dropout is absent at evaluation/export time.
- `metadata.json`: schema version 1, `model_type: mlp_npz`, explicit model version,
  prediction unit `audio_patch`, budget ordering and label protocol provenance.
- `preprocessing.json`: feature names in canonical order, train-only mean/std and
  extractor/schema name `text5-plus-patch-index-v1`.

This patch extractor is intentionally **not registered** in Tony's package.
`deployment_ready` is false and no `policy.json` is generated because calibrated
thresholds and the fallback are unresolved. The export is a numerical handoff,
not a bundle that his current loader can serve. Unit tests verify NPZ/PyTorch
numerical equivalence; they do not claim live patch-level inference works.

Coordinate with Tony on a context-aware patch prediction API and shared causal
feature adapter, then with Yulchan on when the executor can call it per patch.
Include `patch_index` explicitly and version the feature contract. A future
`llm_hidden` feature needs agreed dimensions, availability and inference cost,
a separate schema/model and matching preprocessing. Nothing in the current
baseline silently appends it or changes the serving package.
