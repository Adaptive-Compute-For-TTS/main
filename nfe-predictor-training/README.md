# NFE Predictor Training

Training, evaluation, and experimentation code for the adaptive NFE predictor.

The current baseline predicts NFE pass probabilities per audio patch. Verified
patch labels and an agreed quality protocol are still required before reporting
training results. The 54-prompt full-utterance sweep is a legacy reference.

See [the proposed collection schema and training utilities](docs/patch_dataset.md)
and [the inference compatibility review](docs/inference_compatibility.md).

## CPU setup

Follow [CPU_SETUP.md](CPU_SETUP.md) for the full Windows and Apple Silicon macOS
instructions. Both collaborators should use **Python 3.11** and the pinned
[`requirements-cpu.txt`](requirements-cpu.txt), in a fresh environment inside
this directory.

From the monorepo root, Windows PowerShell:

```powershell
cd nfe-predictor-training
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "pip==25.3"
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements-cpu.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest tests -q
```

On Apple Silicon macOS, with native ARM64 Python 3.11:

```bash
cd nfe-predictor-training
python3.11 -m venv .venv
./.venv/bin/python -m pip install "pip==25.3"
./.venv/bin/python -m pip install --only-binary=:all: -r requirements-cpu.txt
./.venv/bin/python -m pip check
./.venv/bin/python -m pytest tests -q
```

CPU training works independently of GPU inference. The existing baseline takes
the five text features below plus `patch_index` as six inputs and produces nine
NFE pass logits. Use `fit(data, config, device="cpu")` in the existing training
pipeline. Actual patch-level audio collection still requires a separate GPU,
dots.tts, and the WatAI executor integration.

Real experiments require reviewed **`patch-nfe-v2` labels**. Do not train on the
old full-utterance benchmarks or unapproved pilot labels. Pinned versions and
seeds improve reproducibility, but Windows and macOS results may not be
bit-for-bit identical.

## Basic feature extraction

The feature extractor contains inexpensive, deterministic properties that are
available before speech generation:

| Feature | Definition |
| --- | --- |
| `character_count` | All characters, including whitespace |
| `word_count` | Number of whitespace-separated tokens |
| `punctuation_count` | ASCII punctuation characters from `string.punctuation` |
| `digit_count` | Individual numeric characters |
| `average_word_length` | Average token length, including punctuation |

Empty and whitespace-only prompts have an average word length of `0.0`. The
canonical feature ordering is defined by `features.FEATURE_NAMES` and must be
used by both training and inference.

```python
from features import extract_features, features_to_vector

features = extract_features("Hello, world!")
model_input = features_to_vector(features)
```

The optional `context` argument is reserved for future speaker metadata or
embedding features. Basic extraction does not currently use it.

## Feature ablation

`experiments/feature_ablation.py` compares text representations on the same
family-grouped splits and seeds. For each one it reports test Brier score and
loss, budget selection (a threshold chosen on validation, then applied to test:
mean NFE and pass rate against a fixed budget on the same patches), and cost
(input dimensions, extra numbers to ship, per-text extraction time).

Representations are spec strings (`experiments/feature_sets.py`): `text5`,
`rich` (extra handcrafted counts), `a+b` to concatenate, and `pcaK(spec)` for
PCA fitted on training texts only, e.g. `pca4(text5+rich)`. Each one is fitted
on training texts and then frozen, and `patch_index` is appended as in the
baseline.

```bash
./.venv/bin/python -m experiments.feature_ablation --observations reviewed.jsonl --out report.json
./.venv/bin/python -m experiments.feature_ablation --legacy-dry-run   # harness check only
```

Static embeddings (`features/embeddings.py`, model2vec format) need numpy
only: the WordPiece tokenizer is reimplemented and checked against the
`tokenizers` library. Download a model outside the repo and give it a name:

```bash
hf download minishlab/potion-base-8M --local-dir ~/models/potion-base-8M
./.venv/bin/python -m experiments.feature_ablation --legacy-dry-run \
  --static-embeddings potion8m=$HOME/models/potion-base-8M \
  --features "text5,potion8m,pca8(potion8m),text5+pca4(potion8m)"
```

Each model is also registered as `NAME.raw`, which skips the final L2
normalization. Its output is then a plain average of token vectors, so
`pcaK(NAME.raw)` is folded into a `[vocab, K]` table with identical output. For
example, `pca8(potion8m.raw)` ships 0.94 MB instead of the 30 MB full table.
`shipped_params` in the report reflects this.

To run the parity test, set `NFE_STATIC_EMBEDDINGS_DIR` to that directory and
install `tokenizers`; otherwise the test is skipped.

Mean NFE is not the cost on dots.tts's optimized path. Its DiT KV cache only
fits one NFE, so each change of NFE between patches costs about 8× a patch
(see `gpu-executor/docs/design/mixed-nfe-kv-cache.md`). The report therefore
also prices each request's patches in order (`experiments/sequence_cost.py`):

- `--cost-models`: `nfe` (plain mean NFE), `optimized-rtx5060` and
  `eager-rtx5060` (measured per patch and per switch; other NFEs are estimated
  and marked `*`).
- `--policies`: `per_patch`, `min_runR` (switch up at once, down only after R
  patches) and `request_max` (one NFE per request, no switches).

The policy and threshold are chosen on validation as the fastest estimate that
keeps fixed NFE's pass rate on the same patches, then scored once on test.
Patch labels come from one fixed starting state per patch, so sequence pass
rates are estimates.

`--legacy-dry-run` builds provisional request-level labels from the 54-prompt
sweep (`quality-evals/.../clips_arush.csv`). Clips cut off by the length cap
count as unknown, and a clip passes if its WER is at most 0.1. These labels are
never saved, and the report says `"conclusive": false`. Use it to check the
harness, not to compare features.

## Tests

Use the setup/test commands above from this directory. To run just the CPU
training tests, replace `tests` with `tests/test_training.py`. They use synthetic
fixtures and cover training reproducibility, evaluation, and checkpoint and
preprocessing export. The full suite can skip optional audio-scoring integration
tests when the separate scorer dependencies are absent; see
[CPU_SETUP.md](CPU_SETUP.md#cpu-training-checks) for details.
