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

## Tests

Use the setup/test commands above from this directory. To run just the CPU
training tests, replace `tests` with `tests/test_training.py`. They use synthetic
fixtures and cover training reproducibility, evaluation, and checkpoint and
preprocessing export. The full suite can skip optional audio-scoring integration
tests when the separate scorer dependencies are absent; see
[CPU_SETUP.md](CPU_SETUP.md#cpu-training-checks) for details.
