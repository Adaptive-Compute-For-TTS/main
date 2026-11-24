# Controlled patch-level NFE pilot

**Status: collection and provisional labeling tools implemented, not yet run on the dots.tts model.**
This package is an additive patch for `Adaptive-Compute-For-TTS/main` on
`feature/nfe-predictor-training`. Do not replace existing Shivani/Rababb files.
The project is explicitly **patch-level**, not request-level NFE prediction.

## Why new experiments?

The existing 54-prompt sweeps change NFE for the entire speech generation.
They cannot provide counterfactual outcomes for a *single patch*. The archive
also includes continuation-intro contamination and capped long clips.
Reuse the prompt manifest, evaluator and validation modules, but generate new
patch interventions. Existing fixed-NFE results are reference data, not
patch-quality targets.

## What is actually implemented

- `training/pilot.py`: pilot config, early/mid/late patch selection, manifest
  schema, strict matching and grouped provenance validation.
- `training/pilot_capture.py`: the **eager-only** scheduler and a recording
  executor that mirrors the team's eager Euler code. Logs exact hashes of
  pre-patch conditioning, sampled noise and generated patch latent. All earlier
  patches use the reference NFE. One selected patch uses the candidate NFE.
  The hook state is restored after each generation.
- `scripts/generate_patch_pilot.py`: deterministic full reruns, optional speaker
  reference audio, 12-prompt pilot config, restart-safe WAV/trace cache,
  canonical executor equivalence check and `observations.unscored.jsonl`.
- `scripts/score_patch_pilot.py`: uses the team's existing WAV validator and
  per-run WER/CER + optional speaker-sim scorer, paired listening-review CSV,
  and explicitly **unapproved** provisional label policy. Writes both complete
  audited results and a filtered label-ready JSONL.
- `tests/test_patch_pilot.py`: CPU unit tests and fake eager-runtime hook test.

## Pilot size

- 12 real benchmark prompts (6 categories, 1 short + 1 medium each)
- 2 seeds (42, 43), initially 1 voice
- 3 patch positions (early/middle/late)
- 4 candidate NFEs (4, 8, 16, 32), with other patches at reference NFE 32
- 288 candidate observation rows, **up to 240 collection generations**, reusing
  each fixed-reference run for all comparisons at NFE 32, plus one mandatory
  canonical-executor generation for equivalence verification

This is a feasibility pilot, not a final production-sized dataset. The
selected patch positions are determined after the reference run, so very short
outputs may yield fewer positions. A changed patch may cause a later EOS, which
is retained as an unknown measurement if the target itself was never decoded.

## Install and run

1. Clone the monorepo and check out `feature/nfe-predictor-training`.
2. Copy the new files in this package into the matching repo paths.
3. Set up the repo's existing `gpu-executor` environment with **patched**
   `dots.tts` and a suitable CUDA GPU, as documented in
   `gpu-executor/README.md`. The 2B dots.tts model is not installed here.
4. Choose the exact model revision, GPU voice-reference WAV if desired,
   and experiment directory. For reproducibility, use the same conditioning,
   model and dtype across every comparison.

Run from the **monorepo root**. The runtime's upstream
[`from_pretrained` implementation](https://github.com/studio-dots-ai/dots.tts/blob/3cb6f94/src/dots_tts/runtime.py)
supports `revision=`; the collector passes the actual revision and checks the
resolved snapshot directory. `--checkpoint-id` must be `MODEL_REPO@40_HEX_COMMIT`
matching `--model`. A descriptive tag alone is insufficient. The collector
requests deterministic Torch kernels and refuses changed experiment/cache
settings. Its defaults use eager Euler and a 500-patch generation cap.

Use an NVIDIA CUDA host with Python 3.10-3.12 and the existing executor's patched
dots.tts checkout (team source revision `3cb6f94` plus the scheduler/executor
patch). `gpu-executor/scripts/setup_env.sh` installs the team's pinned
Torch/torchaudio 2.8.0 CUDA 12.8 stack. The chosen model must fit that GPU and allow
the experimental sampling overrides; the pilot has not established soar VRAM
requirements. Linux/WSL setup instructions are in `gpu-executor/README.md`.

The following bash variables must refer to real, verified artifacts supplied on
that host. No actual model revision or speaker reference is supplied by this
package. Do not paste placeholder revisions into a real run.

```sh
python nfe-predictor-training/scripts/generate_patch_pilot.py --dry-run

# Set MODEL_REVISION to the verified 40-character model commit and SPEAKER_WAV
# to an existing, validated voice-reference WAV before running these commands.
MODEL_REPO=dots-studio/dots.tts-soar
: "${MODEL_REVISION:?Set the verified model commit}"
: "${SPEAKER_WAV:?Set an existing speaker-reference WAV}"

# Start tiny: 2 observations, 2 collection generations + 1 equivalence generation
python nfe-predictor-training/scripts/generate_patch_pilot.py \
  --model "$MODEL_REPO" --checkpoint-id "$MODEL_REPO@$MODEL_REVISION" \
  --ref-audio "$SPEAKER_WAV" \
  --limit-prompts 1 --limit-seeds 1 --target-patches 1 --budgets 4 32 \
  --out-dir nfe-predictor-training/runs/pilot_smoke

# When the smoke test succeeds, use full config on a NEW output directory:
python nfe-predictor-training/scripts/generate_patch_pilot.py \
  --model "$MODEL_REPO" --checkpoint-id "$MODEL_REPO@$MODEL_REVISION" \
  --ref-audio "$SPEAKER_WAV" \
  --out-dir nfe-predictor-training/runs/pilot_full
```

`--ref-audio` is optional but recommended if speaker similarity is part of the
planned label. The script passes `prompt_text=None` to avoid continuation-mode
prompt leakage. Runtime defaults must allow long enough generation (ideally
500 patches; the script refuses a cap below 100). The script will not run real
inference without a CUDA GPU. It intentionally does not change scheduler,
optimized cache, worker or inference code.

The first reference run checks its float32 sample SHA256, sample rate and patch
schedule against the canonical fixed-NFE team Executor under the same seed.
`equivalence.json` records the result. If they differ, collection stops before
emitting observations. The equivalence check is mandatory and cannot be disabled.
WAV cache checks verify sample hashes and sample rates, not timestamped headers.
Code/configuration fingerprints are recorded, and early-EOS audio/traces are
retained with an unknown target rather than discarded.

## Scoring and provisional labels

Install the shared scorers using the existing worker-compatible requirements.
The standalone scorer requirements pin Torch/NumPy versions that conflict with
each other and with the executor stack; do not install those files over the
executor environment. From the monorepo root, in the executor venv:

```sh
python -m pip install -e wav-validation -c gpu-executor/constraints.txt
python -m pip install -r quality-evals/speaker_sim/requirements-worker.txt -c gpu-executor/constraints.txt

# SAMPLE thresholds only. They are NOT approved and should be reviewed.
python nfe-predictor-training/scripts/score_patch_pilot.py \
  --input nfe-predictor-training/runs/pilot_smoke/observations.unscored.jsonl \
  --speaker-ref "$SPEAKER_WAV" \
  --max-wer 0.10 --max-cer 0.10 \
  --max-wer-delta 0.05 --max-cer-delta 0.03 \
  --max-sim-drop 0.10
```

ASR and WavLM weights load only when real scoring needs them; the first such run
may download them. Speaker-reference hashing must match the generation manifest.
The scorer generates a `listening_review_template.csv`. Review candidate and
matched-reference audio with attention to the target patch's local region and
surrounding speech, then fill `listening_pass` with pass/fail/unknown and rerun
with `--reviews <your_review.csv>`. Keep its experiment, audio and state/noise hash
columns intact: a stale or mismatched review stays unknown. Audio samples and
sample rates are checked against generation records before scoring.
**No default pass/fail labels are emitted
without a listening review.** You can request fully automated *exploratory*
labels using `--auto-provisional`; they remain flagged unapproved.

**Important limitations:** A patch is about 160 ms; WER/CER and WavLM speaker
similarity on full utterances are indirect evidence of this patch's quality.
Their scores may not reflect local artifacts. Do not infer a specific patch's
pronunciation from a whole-WAV WER. For a stronger label, validate with paired
listening and potentially local-window perceptual comparisons. If the reference
itself is invalid or unintelligible, the label stays unknown. Missing/non-finite
scores, unavailable/invalid audio, generation-cap truncation and unverified
equivalence remain unknown, including in automatic exploratory mode. Numeric
scores alone never enter the reviewed-label export.

Outputs:

- `experiment.json`: pinned generation experiment identity
- `equivalence.json`: successful mandatory reference-executor comparison
- `*/reference/{audio.wav,trace.json}`: high-budget common reference
- `*/patch_N/nfe_B/{audio.wav,trace.json}`: one-patch intervention
- `observations.unscored.jsonl`: all observations with **null labels**
- `observations.provisional.jsonl`: scored audit output
- `observations.label_ready.jsonl`: **only** matched, audio-bound human-reviewed
  provisional v1 rows; this legacy name does not imply team approval
- `observations.training_v2.jsonl`: only written on explicit team approval using
  `--approved-protocol-version`; compatible with the existing training pipeline
- `observations.provisional.report.json`: label counts and unapproved protocol
- `listening_review_template.csv`: human review queue

The collector/audit format stays `patch-nfe-v1`. The existing `prepare_data`
requires `patch-nfe-v2`, so do not pass the provisional file directly to training
or change the existing contracts to accept it. Once the team approves the
protocol and the paired reviews are complete, rerun scoring with the same
thresholds, `--reviews <your_review.csv>` and
`--approved-protocol-version <the-actual-approved-version>`. This is an explicit
approval attestation, not an automatic policy decision. The additive
`training.pilot_labels.approved_training_observations` adapter rechecks evidence
and exports v2 metadata/validity fields; auto-only labels cannot be promoted.
No such approval has been performed as part of this integration.

Use `build_targets(read_observations('observations.training_v2.jsonl'))` with the
original nine-budget defaults and `prepare_data` for family-based splitting.
The MLP remains six-input/nine-output. The pilot's other five budgets remain
masked unknown; an unmeasured output head has no direct training signal.

## Leakage and quality gates

- Split by prompt family, not WAV, patch index, seed or NFE. The 54-prompt
  historical split is **not** a final split for a newly expanded corpus.
- Store model revision, prompt text, reference voice hash, seed, Euler solver,
  guidance, speaker scale, patch index, previous schedule, input-state and
  sampled-noise hashes for every observation.
- Reject any comparison with different target **incoming state or noise**, or
  different earlier patch outputs; never silently assign it a `False` label.
- Preserve fallback and pass-quality policy versions separately; Eric owns
  final confidence thresholds and fallback.
- Training and inference feature preprocessing must match; Tony's existing
  inference package is request-level only and needs patch interface changes.
- The optimized path has timestep-specific KV caches; NFE switching savings
  are **not demonstrated** by this eager-mode data-collection tool.

## Validation

```sh
cd nfe-predictor-training
python -m pytest tests/test_patch_pilot.py -q
```

Tests are CPU-only and use a small fake runtime for hook integration. They do
**not** run the real dots.tts model, establish actual quality, or measure speed.
Additional scorer tests reuse the real WAV validator and quality evaluator but
inject synthetic ASR transcripts and speaker similarities; their temporary WAVs
and labels are unit-test fixtures, not collected observations. Those tests need
`soundfile`, `wav-validation` and the WER dependencies; they skip when unavailable.
The root `--dry-run` needs only Python, loads no model and writes no outputs.

## Relevant repo code

- `gpu-executor/src/adaptive_nfe/executor.py`
- `gpu-executor/src/adaptive_nfe/scheduler.py`
- `gpu-executor/patches/dots_tts_scheduler_hook.patch`
- `quality-evals/speaker_sim/evaluate_run.py`
- `wav-validation/src/wav_validation/checks.py`
- `nfe-adapter/predictor` on branch `nfe-predictor-inference`
