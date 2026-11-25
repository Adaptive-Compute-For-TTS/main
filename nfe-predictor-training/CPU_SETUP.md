# Reproducible CPU setup

Rababb and Shivani should both use **CPython 3.11** and
[`requirements-cpu.txt`](requirements-cpu.txt) on
`feature/nfe-predictor-training`. Supported targets are Windows x86-64 and
Apple Silicon macOS with a native ARM64 Python interpreter. Use the same Python
patch version when comparing experiments; Windows validation uses Python 3.11.0.

This environment runs the existing six-input, nine-output MLP, feature
normalization, masked BCE loss, evaluation, and checkpoint/preprocessing export
on CPU. It does not install dots.tts, CUDA, or the GPU executor. Actual
patch-level audio collection still requires the separate GPU environment.

## Get the branch

For a new clone, run these commands from the folder where you keep repositories:

```text
git clone --branch feature/nfe-predictor-training https://github.com/Adaptive-Compute-For-TTS/main.git
cd main
```

For an existing clone, inspect `git status` and preserve any local work before
switching branches or pulling. With a clean working tree:

```text
git switch feature/nfe-predictor-training
git pull --ff-only origin feature/nfe-predictor-training
```

Use a fresh `.venv` **inside `nfe-predictor-training/`**, separate from the
monorepo's or GPU executor's environments. If that environment already exists,
use a fresh clone for a clean validation instead of mixing installations.
Activation is unnecessary; the commands below select the interpreter explicitly.

## Windows (PowerShell)

Install 64-bit Python 3.11 with the Windows Python launcher first. From the
monorepo root:

```powershell
cd nfe-predictor-training
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "pip==25.3"
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements-cpu.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest tests -q
```

Check the CPU build and all six training imports from the same directory:

```powershell
.\.venv\Scripts\python.exe -c "import sys, torch; print(sys.version); print(torch.__version__); assert torch.version.cuda is None; assert not torch.cuda.is_available()"
.\.venv\Scripts\python.exe -c "import training.train, training.dataset, training.targets, training.evaluate, models.mlp, features.extractor; print('All six imports passed')"
```

The expected Torch version is `2.8.0+cpu`: `+cpu` identifies the official
Windows CPU build of the pinned PyTorch 2.8.0 release. The requirements file
selects that wheel directly and checks its SHA-256 hash.

## macOS (Apple Silicon)

Install native ARM64 Python 3.11 first. Avoid an Intel Python running under
Rosetta. From the monorepo root:

```bash
cd nfe-predictor-training
python3.11 --version
python3.11 -c "import platform; print(platform.machine()); assert platform.machine() == 'arm64'"
python3.11 -m venv .venv
./.venv/bin/python -m pip install "pip==25.3"
./.venv/bin/python -m pip install --only-binary=:all: -r requirements-cpu.txt
./.venv/bin/python -m pip check
./.venv/bin/python -m pytest tests -q
./.venv/bin/python -c "import torch; print(torch.__version__); assert torch.version.cuda is None; assert not torch.cuda.is_available()"
./.venv/bin/python -c "import training.train, training.dataset, training.targets, training.evaluate, models.mlp, features.extractor; print('All six imports passed')"
```

The expected Torch version is `2.8.0`. This wheel has no CUDA dependency.
Apple's MPS backend may be available, but use `device="cpu"` for these experiments.
The shared requirements use platform markers to select the correct Torch wheel;
all other runtime/test dependencies are pinned too, with Colorama needed only
on Windows. No packages need to be compiled from source.

Python 3.11 wheels are published for Apple Silicon by
[PyTorch 2.8.0](https://pypi.org/project/torch/2.8.0/#files),
[NumPy 2.2.6](https://pypi.org/project/numpy/2.2.6/#files), and
[SoundFile 0.13.1](https://pypi.org/project/soundfile/0.13.1/#files).
[pytest 8.4.2](https://pypi.org/project/pytest/8.4.2/#files) uses a platform-independent
wheel. The native wheels for these packages include macOS 11+ ARM64 builds.
Wheel availability is a compatibility review, **not a successful Mac test run**.

## CPU training checks

Run the existing synthetic training tests separately when checking the pipeline:

```powershell
# Windows, from nfe-predictor-training/
.\.venv\Scripts\python.exe -m pytest tests/test_training.py -q
```

```bash
# macOS, from nfe-predictor-training/
./.venv/bin/python -m pytest tests/test_training.py -q
```

These tests exercise seeded training and loader shuffling, the six-input,
nine-output model, masked loss, evaluation, best-validation selection, and
checkpoint/preprocessing/NumPy export round trips. The dataset tests also check
normalization and split isolation. Test fixtures are synthetic; they are not
scientific training results or approved pilot labels.

The full suite also includes optional scorer integration tests. They use
`pytest.importorskip` for `wav_validation`, `faster_whisper`, `jiwer`,
`whisper_normalizer`, and `pandas`, which belong to the separate audio
validation/scoring environment. Those tests can skip in this minimal CPU
training environment. Inspect the reasons with `python -m pytest tests -q -rs`
using your virtual environment's interpreter. Core training tests must pass.
SoundFile is included so the CPU WAV-cache test runs rather than skips.

## Running real experiments

Reuse `training.train.prepare_data`, `TrainingConfig`, `fit`, and
`save_artifacts`, plus `training.evaluate.evaluate`. The existing
[training/data documentation](docs/patch_dataset.md#training-and-evaluation-foundations)
shows how to assemble an experiment; use `fit(data, config, device="cpu")`.

Real training requires reviewed **`patch-nfe-v2` labels** with the agreed
labeling protocol and provenance. Do not train on the old full-utterance NFE
benchmarks, planning examples, or unapproved provisional pilot labels. Actual
patch-level audio collection uses dots.tts and the WatAI executor in a separate
GPU environment; its dependencies do not belong in this CPU environment.

For a reproducible comparison, keep the branch commit, reviewed dataset and its
hash, grouped split assignments, seed, hyperparameters, preprocessing, and model
architecture fixed. `seed_everything` and the existing artifact export preserve
the relevant training metadata. Save `python --version` and `python -m pip freeze`
with each run using the virtual environment's interpreter. Use an empty artifact
directory so earlier experiments are preserved.

Pinned packages and seeds do not guarantee bit-for-bit identical results across
Windows and macOS: hardware and numerical libraries can change floating-point
results. Compare metrics with appropriate tolerances.

## Validation status (2026-11-25)

The Windows commands above were executed in a fresh, isolated virtual
environment with these installed versions:

| Component | Version |
| --- | --- |
| CPython (Windows x86-64) | 3.11.0 |
| PyTorch | 2.8.0+cpu |
| NumPy | 2.2.6 |
| pytest | 8.4.2 |
| SoundFile | 0.13.1 |
| pip | 25.3 |
| setuptools | 80.9.0 |

All supporting packages match the exact pins in `requirements-cpu.txt`.
`pip check` passed, all six training-module imports passed, and Torch reported
no CUDA build and no available CUDA device.

- Full suite: **123 passed, 0 failed, 2 skipped**.
- CPU training suite (`tests/test_training.py`): **13 passed, 0 failed, 0 skipped**;
  these are included in the full-suite total.
- The two skips were `test_scoring_review_and_approved_v2_handoff` and
  `test_auto_scores_are_not_label_ready_and_changed_audio_stays_unknown`, both
  because `wav_validation` is absent. The other scorer dependencies listed above
  are also intentionally outside this minimal environment.
- Apple Silicon: compatible wheel tags and transitive version constraints were
  reviewed for all 20 applicable pinned packages. **No tests were executed on a
  Mac.** Mac compatibility is reviewed, not runtime-validated.

No ML implementation or tests were changed, and no real-data training or audio
collection was performed.
