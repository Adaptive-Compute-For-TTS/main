# gpu-workers

The long-running process on the GPU machine that turns queued runs into WAVs. The backend (FastAPI) only records a run as `queued` in SQLite; the worker picks it up, generates the audio with `gpu-executor/` at the requested NFE, and writes the result back. The frontend polls `GET /runs/{id}` until it is `completed` or `failed`.

Owners: Tony, Nick Kim, Eric.

Status: the worker core (`src/gpu_worker/`) runs one job end to end: generate, validate, store, build the row. Not yet connected to the queue; the backend still uses its placeholder worker (`backend/app/workers/local_worker.py`).

## Flow

Follows the system diagram:

```
React -> FastAPI POST /runs -> job queue (SQLite row, status=queued)
                                  |
GPU worker:  claim run (status=running)
             -> NFE adapter + fixed-NFE executor     generate_fixed_nfe(rt, nfe, text=prompt)   gpu-executor/
             -> local GPU execution                  dots.tts on this machine's GPU
             -> runtime telemetry (Niranjan)         time, RTF, peak VRAM, ...
             -> WAV validation (Nick)                validate_and_store(...)               wav-validation/
                  passed: <storage>/<run_id>/audio.wav     failed: _rejected/<run_id>/, run fails
             -> quality eval (Arush, Atharv)         passed clips only
             -> local WAV storage + SQLite (Sai)     status=completed + result + audio_path
                                                     (or status=failed + error_message)
                                  |
FastAPI results -> React dashboard
```

Check first, then store: the worker never saves the WAV itself. It passes the audio straight from the executor to `validate_and_store`, which checks it and writes the file only to the right place (`<storage>/<run_id>/` if it passed, `_rejected/<run_id>/` if not).

The model is loaded once when the worker starts (plus a warmup), then reused for every run.

## Calling it (for the queue loop)

```python
from gpu_worker import GpuWorker, Job, WorkerConfig

worker = GpuWorker.load(WorkerConfig.from_env())   # once at startup: loads dots.tts, warms up
outcome = worker.run_job(Job(run.id, run.prompt, run.nfe))   # per claimed run; never raises for a failed run
# outcome.status ("completed" / "failed"), outcome.result, outcome.audio_path, outcome.error_message
#   -> repository.mark_run_completed(db, run.id, outcome.result, audio_path=outcome.audio_path)
#   or repository.mark_run_failed(db, run.id, outcome.error_message)
```

Settings (model, `optimize`, seed, storage folder, warmup NFEs, ...) come from `GPU_WORKER_*` environment variables; see `src/gpu_worker/config.py`. Defaults match `gpu-executor/scripts/generate.py`: soar, bf16, eager, seed 42, guidance 1.2, speaker scale 1.5. Every run is seeded the same, so the same prompt and NFE give the same audio.

## Setup and checks

Needs the gpu-executor setup (dots.tts clone with the hook patch next to the monorepo, see `gpu-executor/README.md`). The worker shares gpu-executor's venv, so it runs exactly the executor code gpu-executor's checks test.

```bash
bash gpu-workers/scripts/setup_env.sh && source gpu-executor/.venv/bin/activate
cd gpu-workers
pytest tests/                                                    # fake runtime: no weights, no GPU
python scripts/run_one.py --prompt "Hello there." --nfe 10       # real model on this GPU, prints the row
```

## Folders

- `src/gpu_worker/`: `config.py` (settings), `worker.py` (`GpuWorker`: load once, `run_job`)
- `scripts/`: `setup_env.sh`, `run_one.py` (one job by hand)
- `tests/`: fake runtime driving the real `generate_fixed_nfe` and `validate_and_store`
- `runs/`: default storage folder, gitignored

## Interface (draft, to agree with backend)

In, from the `runs` table: `id`, `prompt`, `nfe` (4-32).

Out, written to the same row:
- `audio_path`: `<storage>/<run_id>/audio.wav`, from `validate_and_store` (only for clips that passed)
- `result` (JSON): `nfe`, `nfe_per_patch`, `num_patches`, `executor_used_all_patches`, `sample_rate`, `audio_sha256`, the settings (`model`, `precision`, `optimize`, `seed`, `ode_method`, `guidance_scale`, `speaker_scale`), `gpu`, `worker_version`, plus `validation` (what `validate_and_store` returns), `metrics` and `quality`, and `telemetry` (empty for now). Also kept for failed validations (then `quality` is empty), for debugging.
- on error: `status=failed`, `error_message` (out of GPU memory, NFE check failed, WAV failed validation with its flags, ...)

## Quality eval

Runs inside the worker, on the stored clip, before the run is marked completed (agreed with Sai). `GpuWorker.load` builds one `QualityEvaluator` (quality-evals/speaker_sim/evaluate_run.py; its models load on the first scored clip, so the first run is slower and needs the model files cached or downloadable; `time_s` and `rtf` are unaffected). Clips that failed validation are not scored. The evaluator never raises, so a scorer that fails leaves its value `None` plus a message in `quality["errors"]` and the run still completes.

- `result["metrics"]`: `audio_s`, `time_s`, `rtf`, `peak_vram_mb` (generation only, quality eval is not included)
- `result["quality"]`: `scored`, `sim_ref`, `wer`, `cer`, `transcript`, `sim_model`, `asr_model`, `errors`
- `GPU_WORKER_REFERENCE_WAV`: the voice `sim_ref` compares against. The worker generates without a prompt voice, so unset means `sim_ref` is `None` (WER/CER still work). Still to agree which voice.
- Install on top of the constraints: `pip install -r quality-evals/speaker_sim/requirements-worker.txt -c gpu-executor/constraints.txt`

## Open questions

- Separate worker process polling SQLite (proposed: survives restarts, keeps dots.tts out of the backend env) vs. running inside FastAPI
- Eager or optimized path for the demo (optimized is ~2x faster but compiles ~3 min at startup)
- Model (soar or base) and where WAVs are stored (`storage_dir` is shared by wav-validation and Sai's storage)
- Retrying a run: `validate_and_store` refuses to store the same `run_id` twice, even after a failed validation, so a retry needs a new run id or a backend rule

## Plan

- worker core: load model, run one job, validate + store, build the row (Tony) - first version done
- claiming jobs, polling, recovering runs stuck in `running` (Eric)
- WAV validation step + testing on the GPU (Nick)
- end-to-end with the backend by Nov 15
