# gpu-workers

The long-running process on the GPU machine that turns queued runs into WAVs. The backend (FastAPI) only records a run as `queued` in SQLite; the worker picks it up, generates the audio with `gpu-executor/` at the requested NFE, and writes the result back. The frontend polls `GET /runs/{id}` until it is `completed` or `failed`.

Owners: Tony, Nick Kim, Eric.

Status: planning. Nothing runs yet; the backend still uses its placeholder worker (`backend/app/workers/local_worker.py`).

## Flow

```
POST /runs  ->  SQLite row, status=queued
                  |
GPU worker:  claim run (status=running)
             -> generate_fixed_nfe(rt, nfe, text=prompt)      gpu-executor/
             -> write <storage>/<run_id>/audio.wav
             -> WAV validation (Nick), telemetry (Niranjan)
             -> status=completed + result + audio_path   (or status=failed + error_message)
```

The model is loaded once when the worker starts (plus a warmup), then reused for every run.

## Interface (draft, to agree with backend)

In, from the `runs` table: `id`, `prompt`, `nfe` (4-32).

Out, written to the same row:
- `audio_path`: `<storage>/<run_id>/audio.wav`
- `result` (JSON): `nfe_per_patch`, `num_patches`, `audio_s`, `time_s`, `rtf`, `peak_vram_mb`, `audio_sha256`, `model`, `optimize`, `seed`, plus `validation` and `telemetry` sections
- on error: `status=failed`, `error_message` (out of GPU memory, NFE check failed, WAV failed validation, ...)

## Open questions

- Separate worker process polling SQLite (proposed: survives restarts, keeps dots.tts out of the backend env) vs. running inside FastAPI
- Eager or optimized path for the demo (optimized is ~2x faster but compiles ~3 min at startup)
- Model (soar or base) and where WAVs are stored

## Plan

- worker core: load model, run one job, write WAV + result (Tony)
- claiming jobs, polling, recovering runs stuck in `running` (Eric)
- WAV validation step + testing on the GPU (Nick)
- end-to-end with the backend by Nov 15
