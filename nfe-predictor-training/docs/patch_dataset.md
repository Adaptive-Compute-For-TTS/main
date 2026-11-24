# Proposed patch experiment contract

This is preparation for data collection, not an approved quality policy or a
collected dataset. Full-utterance fixed-NFE WER/CER/speaker-similarity results
must not become ground-truth patch labels. The existing 54-prompt CSV remains a
legacy benchmark reference; new training does not use it as its split.

## Observation schema

One UTF-8 JSON object per line represents one `(comparison_id, nfe)` measurement.
The Python schema/validator lives in `training.contracts.validate_observation`.
`patch-nfe-v2` adds strict collection fields; existing `patch-nfe-v1` remains
readable for compatibility. The new training preparation utility requires v2.

| Field | Meaning and validation |
| --- | --- |
| `schema_version` | `patch-nfe-v2` for new collection |
| `feature_schema_version` | `text5-plus-patch-index-v1` |
| `record_status` | `planned` or `measured`; planned records have no quality results |
| `experiment_id`, `comparison_id` | Nonempty IDs; comparison IDs must be unique across the input dataset |
| `prompt_id`, `prompt_family_id` | Stable prompt and reviewed family IDs; a prompt ID has one text and family |
| `speaker_id` | Stable identity of the conditioning speaker |
| `text` | Prompt text available before generation; never a generated transcript |
| `patch_id`, `patch_index` | Stable patch ID and zero-based nonnegative integer index |
| `nfe` | Positive integer on the configured experimental budget axis |
| `seed` | Integer generation seed, fixed within a controlled comparison |
| `checkpoint_id` | Exact model revision, not a moving alias |
| `conditioning_id` | Immutable ID/digest resolving to the complete conditioning configuration |
| `solver` | Versioned solver/configuration identifier, fixed within a comparison |
| `anchor_state_hash` | Hash identifying the exact state immediately preceding this patch |
| `noise_hash` | Hash identifying the exact initial noise for this patch |
| `pre_generation_features` | Optional mapping of named finite scalars; retained as metadata only |
| `quality_measurements` | Mapping of scorer names to finite numeric measurements or `null`; may be empty |
| `quality_pass` | Explicit reviewed `true`, `false`, or `null` (unknown) |
| `label_protocol` | Versioned review/scoring protocol; required for observed labels, otherwise nullable |
| `audio_valid`, `not_truncated`, `counterfactual_reviewed` | Required boolean-or-null flags; known for observed labels |

Known labels require a measured, reviewed comparison. A positive label cannot
contradict invalid/truncated audio. A reviewed negative may include invalid audio.
Flags and numeric measurements never automatically imply pass or failure; only
an explicit label under an agreed protocol creates a supervised target.
This provenance check does not prescribe numeric quality thresholds.

`PatchTarget.observations` retains the original per-budget records, including all
quality measurements and generation metadata. The target and mask axes follow
`PatchTarget.budgets`; absent budgets and unknown labels have mask zero. Only a
complete set of observed failures sets `all_tested_failed`. Unknown-only patches
cannot enter supervised training.

## Controlled collection procedure (requires executor-team implementation)

1. Agree with Nick/Yulchan on patch boundaries, zero-based indexing, the meaning
   of the conditioning configuration, checkpoint pinning and state serialization.
2. Capture the entire preceding generation state and initial noise once, before
   the chosen patch. Hashes must cover the actual tensors, shapes, dtypes and
   relevant generator/cache state using an agreed serialization/hash version.
3. Restore that identical snapshot and noise separately for each candidate NFE.
   Reusing a seed alone does not prove identical noise or preceding state.
4. Record each budget independently. Changing state, noise, seed, speaker,
   conditioning, checkpoint, solver/configuration, prompt or patch means a new
   comparison ID. The validator rejects mixed contexts and duplicate budgets.
5. Have Arush/Atharv define and version patch evaluation and attribution. Record
   scorer measurements and audio-validity flags, then reviewed pass/fail labels.
   Measurements without an agreed outcome remain `null` labels.

`configs/patch_experiment_example.jsonl` is an uncollected planning example with
three budgets and **no quality labels or measurements**. Its placeholder IDs and
hashes are not real replay evidence. Replace them with captured provenance; do
not turn them into labels. The validator cannot verify audio, hash contents,
causal feature availability, replay fidelity or scientific label validity.
No GPU replay, executor, scheduler or teammate implementation is changed here.

## Training and evaluation foundations

`prepare_data` groups by reviewed prompt family across every comparison,
including unlabeled ones. Every patch, speaker, seed and NFE for that family stays
in one split. Ratios apply to family counts and are not a promise of exact prompt
counts or stratification. Audit family membership and coverage before reporting
results. Externally supplied split assignments are also checked for leakage.
The helper refuses empty labeled train, validation or test partitions and mixed
observed protocol versions. Select a consistent protocol before combining runs.

After splitting, unknown-only patches are counted and excluded. The normalizer
fits population means/stds on labeled **training examples only**; constant
features use scale 1. The same frozen transform is applied to validation/test.
Normalization weights each patch comparison equally, not each NFE row. Fit IDs,
splits and a source-record hash are saved with artifacts for audit. Supplied split
assignments are identified separately rather than attributed to the default seed.

The baseline still uses the five values in `features.FEATURE_NAMES`, followed by
`patch_index`, with hidden layers 64 and 32 and nine independent logits by
default. Additional collected metadata does not enter the model. There are no
LLM hidden-state inputs. State, speaker and noise are controlled label context,
not features the current model can distinguish. Test robustness across those
conditions before interpreting its probabilities.

The functions below are callable by `experiments/` and Eason's future runner.
The current repository only has a placeholder `experiments/feature_ablation.py`;
these utilities do not create a competing experiment framework or CLI.

```python
from training.targets import read_observations, build_targets
from training.train import TrainingConfig, prepare_data, fit, make_loader, save_artifacts
from training.evaluate import evaluate

# Use a real, reviewed patch dataset, not the planning example or utterance CSV.
examples = build_targets(read_observations("reviewed_patch_observations.jsonl"))
data = prepare_data(examples, seed=42)
config = TrainingConfig(seed=42)  # experiment defaults must be reviewed/tuned
result = fit(data, config)
save_artifacts(result, data, config, "new_empty_run_directory", model_version="your-run-id")
# Access test data only after model/experiment choices are fixed.
test_report = evaluate(result.model, make_loader(data.datasets["test"]), data.budgets)
```

The loop uses masked BCE and observation-weighted validation loss, saves the best
validation epoch and does not touch test examples during fitting. Reproducibility
seeds Python, NumPy, Torch, loader shuffles and workers, and requests deterministic
kernels. Cross-version/hardware bitwise equality is not promised. For GPU work,
set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before initializing CUDA; deterministic
kernels may be unavailable and then Torch should raise. CPU unit tests exercise
the training path using synthetic fixtures only.

`evaluate` reports observed-label loss and per-budget support, confusion counts,
accuracy, precision, recall and Brier score. Its default classification cutoff
0.5 is diagnostic, not an approved quality or budget-selection threshold.
Undefined metrics are `None`. `fixed_nfe_baseline` reports observed pass rate and
coverage for a caller-selected fixed budget. `compare_fixed_nfe` requires an
explicit budget-selection threshold, fallback and fixed budget; it reports both
policies on the same patches where both chosen outcomes were measured. Partial
coverage can introduce selection bias. Mean NFE is a compute proxy, not measured
latency. These offline controlled-patch comparisons do not validate a sequence
of adaptively generated patches.

`save_artifacts` writes `checkpoint.pt`, `metadata.json`, `preprocessing.json` and
`model.npz` into an empty directory. It saves the actual architecture, preprocessing,
best epoch, history, seed/hyperparameters, source hash, label protocol, split IDs,
per-split observed support for every budget, caller experiment configuration and
dependency versions. A budget with no training observations has no direct
supervised training signal, even when other budgets have labels. `load_checkpoint`
restores an evaluation model and preprocessing. Optimizer resume is not provided.
No model artifacts or synthetic benchmark results are checked into the repository.

## Open decisions for the team

- Nick/Yulchan: patch identity/boundaries, exact snapshot/noise capture and replay,
  versioned hash/config formats, and availability of causal patch features.
- Arush/Atharv: local-patch versus continuation quality evaluation, pass/fail
  attribution, audio checks, protocol approval and calibrated numeric thresholds.
- Eason: runner integration, experimental configs and selection methodology.
- Tony/Eric: patch-context inference API, calibrated selection thresholds,
  candidate budget subset and fallback behavior. Current budgets and fallback 32
  remain experimental proposals, without a quality guarantee.
- Future `llm_hidden`: dimensions, availability, serialization and inference cost
  must be agreed before a separate feature-schema/model experiment.
