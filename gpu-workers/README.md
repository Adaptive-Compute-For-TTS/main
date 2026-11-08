# gpu-workers

The long-running process on the GPU machine that turns queued runs into WAVs. The backend (FastAPI) only records a run as `queued` in SQLite; the worker picks it up, generates the audio with `gpu-executor/` at the requested NFE, and writes the result back. The frontend polls `GET /runs/{id}` until it is `completed` or `failed`.

Owners: Tony, Nick Kim, Eric.

Status: planning. Nothing runs yet; the backend still uses its placeholder worker (`backend/app/workers/local_worker.py`).

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

## Interface (draft, to agree with backend)

In, from the `runs` table: `id`, `prompt`, `nfe` (4-32).

Out, written to the same row:
- `audio_path`: `<storage>/<run_id>/audio.wav`, from `validate_and_store` (only for clips that passed)
- `result` (JSON): `nfe_per_patch`, `num_patches`, `audio_s`, `time_s`, `rtf`, `peak_vram_mb`, `audio_sha256`, `model`, `optimize`, `seed`, plus `telemetry`, `validation` (what `validate_and_store` returns) and `quality` sections
- on error: `status=failed`, `error_message` (out of GPU memory, NFE check failed, WAV failed validation with its flags, ...)

## Open questions

- Separate worker process polling SQLite (proposed: survives restarts, keeps dots.tts out of the backend env) vs. running inside FastAPI
- Eager or optimized path for the demo (optimized is ~2x faster but compiles ~3 min at startup)
- Model (soar or base) and where WAVs are stored (`storage_dir` is shared by wav-validation and Sai's storage)
- Retrying a run: `validate_and_store` refuses to store the same `run_id` twice, even after a failed validation, so a retry needs a new run id or a backend rule
- Does quality eval run inside the worker, or afterwards on stored clips?

## Plan

- worker core: load model, run one job, write WAV + result (Tony)
- claiming jobs, polling, recovering runs stuck in `running` (Eric)
- WAV validation step + testing on the GPU (Nick)
- end-to-end with the backend by Nov 15
