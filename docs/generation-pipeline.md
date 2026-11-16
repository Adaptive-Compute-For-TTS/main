# Generation pipeline

The same request ID follows this path:

1. The browser submits its text and fixed NFE through Next.js `POST /api/runs`.
2. FastAPI `POST /runs` commits a `queued` SQLite row, then adds its UUID to `queue.Queue`.
3. With `ADAPTIVE_TTS_WORKER_MODE=gpu`, one consumer claims the row as `running`. `GpuProcessor` loads `GpuWorker` once and reuses it across requests.
4. `GpuWorker.run_job(Job(id, prompt, nfe))` calls the existing fixed-NFE executor. The executor checks that every generated patch used the selected NFE. WAV validation/storage remains owned by `wav-validation`.
5. The consumer measures generation with the team's telemetry harness, validates/stores the WAV, then runs the team's per-run quality evaluator on accepted clips. It saves the complete outcome/result/audio path and marks the row `completed` or `failed`. Exceptions become failed runs and the consumer continues to the next job. Optional scorer failures retain completed audio and expose diagnostic errors; failed WAV validation retains its measurements and skips quality scoring.
6. The browser's Your runs table polls its own active IDs every two seconds through the Next.js proxy. Its current-progress column displays queued, generating audio, completed or failed. The API has no SSE broadcast or granular worker progress yet. Completed WAVs are downloaded through FastAPI `GET /runs/{id}/audio` via Next.js, committed to IndexedDB, then played using a local blob URL in the selected run's details.

The backend keeps SQLite records and original WAVs on its machine. The browser separately stores full run metadata (including all result metrics) in localStorage and WAV blobs in IndexedDB, per browser/origin, without an arbitrary history-count cap. It does not import every run from the database. Cached audio and details survive reload with the backend offline. Browser quota/availability failures are shown, and clearing site data removes local copies; Download WAV and Export run details provide file copies. Removing a local run deletes its browser metadata and blob, without deleting server records/files. If metadata storage is unavailable, details remain in memory for the tab.

## Worker environment

For Apple Silicon setup on another checkout, run `bash scripts/setup_generation_mac.sh`. It installs only beneath `.local/`, uses pinned dots.tts plus the team hook patch, and does not create runs or download model weights until a request is submitted.

The API and queue consumer must run in the **same Python process and environment**. Install backend requirements into the environment that already has patched dots.tts, adaptive-nfe, gpu-workers and wav-validation. A separate GPU worker process cannot share this Python queue by importing it. Run one Uvicorn worker.

`queue-only` remains available for inspecting submissions without generation. `placeholder` exists solely for legacy tests and does not produce audio; the normal launcher does not offer it.

GPU environment command:

```sh
python3 scripts/dev.py --worker-mode gpu --python /absolute/path/to/gpu-env/bin/python
```

Validated audio is stored beneath `backend/data/audio` by default, ignored by Git. `ADAPTIVE_TTS_AUDIO_DIR` configures another location. The API refuses paths outside that root and does not serve failed/uncompleted runs.

## MacBook test plan

This Mac is Apple Silicon. CUDA is unavailable. The checked-out dots.tts runtime selects CPU when CUDA is absent; CPU execution requires `float32` and `optimize=False`. It does not select MPS. This is a slow correctness path, not a CUDA performance benchmark.

Use native ARM64 Python 3.12. The system's `/usr/local` Python 3.12 is Intel; the team-pinned PyTorch 2.8 has no macOS Intel wheel. A project-local native environment avoids changing the existing API environment. Pynini requires OpenFst; its official Mac installation path is conda-forge.

Current local preparation uses micromamba at `.local/bin/micromamba`, a conda-forge Python/Pynini environment at `.local/tts-env`, and a separate patched dots.tts clone at `.local/dots.tts`. The original AdaptiveCompute project's dots.tts checkout is not modified.

```sh
MAMBA_ROOT_PREFIX="$PWD/.local/mamba" .local/bin/micromamba create -y \
  -p "$PWD/.local/tts-env" -c conda-forge --override-channels \
  python=3.12 pynini=2.1.7 pip
uv pip install --python .local/tts-env/bin/python \
  -r backend/requirements.txt -e .local/dots.tts \
  -e gpu-executor -e gpu-workers -e wav-validation \
  -c gpu-executor/constraints.txt
GPU_WORKER_PRECISION=float32 GPU_WORKER_OPTIMIZE=false \
  GPU_WORKER_WARMUP_NFES='' \
  python3 scripts/dev.py --worker-mode gpu --python .local/tts-env/bin/python
```

For a first real-generation check, use a temporary `ADAPTIVE_TTS_DATABASE_URL` and `ADAPTIVE_TTS_AUDIO_DIR`, a short prompt, and NFE 4. Verify the stored UUID, running/completed states, per-patch executor NFE, validated WAV, successful audio response and browser playback. Then verify browser refresh restores that receipt. Repeat with NFE 20/32 after confirming runtime and memory costs. Keep test requests out of the development database.

## Verification boundaries

Backend tests inject a processor that writes an explicitly identified WAV fixture. They verify queue consumption, SQLite outcomes, file serving, range requests, failure recovery and path confinement; they do not prove real dots.tts inference or perceptual speech quality. Frontend proxy tests verify binary forwarding. A real-model run must be recorded separately before claiming the entire generation objective complete.

Official dependency instructions:

- [Pynini](https://github.com/kylebgorman/pynini#installation-instructions)
- [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)

See [SQLite schema](sqlite-schema.md) for the current fields, constraints, result format and queue boundaries.

## Per-run diagnostics

`backend/app/workers/diagnostics.py` adapts `gpu-telemetry/telemetry` and `quality-evals/speaker_sim/evaluate_run.py` to the worker. Telemetry covers exactly one generation, with no extra warmups/repeats. The fingerprint, timer, peak allocated/reserved memory and NVML sampling stop before WAV validation and ASR. CPU timing and runtime metadata work on a Mac; NVIDIA-only measurements are explicitly unavailable. The core worker accepts optional adapters so unit tests and standalone uses need no ASR download.

Install diagnostics dependencies in the same worker environment, using the executor constraints rather than the telemetry harness's standalone torch pins:

```sh
uv pip install --python .local/tts-env/bin/python \
  -r quality-evals/speaker_sim/requirements-worker.txt nvidia-ml-py \
  -c gpu-executor/constraints.txt
```

Both worker setup scripts include these dependencies. No GPU is required for WER/CER. The team's default is `faster-whisper-small`, CPU/int8, English, with the existing normalization and intro-adjusted, capped scores. The first scoring request downloads the ASR model, and later requests reuse it. `ADAPTIVE_TTS_ASR_MODEL` changes the Whisper model; the recorded settings identify the scorer. `ADAPTIVE_TTS_QUALITY_REFERENCE_WAV=/absolute/path/to/reference.wav` enables speaker similarity against a real reference recording. Without it, similarity is skipped with an explicit note; WER/CER still run. Model downloads or scorer errors appear in the quality result without losing validated audio.

The runs table calls active work **Processing**, including generation, validation and scoring. Historical records are not fabricated or backfilled: WAV validation already present in older results is shown, but missing telemetry and quality require a new run. The UI displays validation pass/fail/cautions and signal checks, CPU/GPU telemetry, transcript, WER/CER and similarity availability. It preserves the complete result locally alongside the cached WAV.
