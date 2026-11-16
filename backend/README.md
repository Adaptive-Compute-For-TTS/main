# Adaptive Compute TTS Backend

Local FastAPI backend for submitting TTS runs, storing their state in SQLite,
and exposing results to the React frontend.

The default mode is **queue-only**. Submitted jobs remain queued until a real
GPU processor is connected. The backend stores each request in SQLite and adds
its UUID to an actual Python `queue.Queue` in this process. It does not generate
audio or mark a run successful by itself.

## Current flow

1. `POST /runs` validates a prompt and an NFE value from 4 through 32.
2. The backend creates a SQLite record with status `queued`.
3. The backend enqueues the committed record's UUID.
4. `GET /queue` reports queued UUIDs without consuming them.
5. `GET /runs/{id}` retrieves the durable request and its current state.

`POST /runs` returns HTTP 202 and the existing run representation, including
`id`, `prompt`, `nfe`, and `status: "queued"`. NFE accepts integers from 4 through
32. Prompt text is trimmed and must contain 1–5000 characters. Invalid requests
return 422 without creating a database record or queue entry.

No demonstration records are seeded. On startup, queued SQLite rows are restored
to the in-memory queue in creation order. Repeated startup on the same worker
does not duplicate entries. Rows left `running` by an interrupted process become
`failed` with an explanation; they are not replayed automatically.

## Requirements

- Python 3.9 or newer
- A local virtual environment is recommended

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
```

## Run locally

From the repository root:

```bash
python -m uvicorn backend.app.main:app --reload
```

The API is available at:

- API: <http://127.0.0.1:8000>
- Interactive documentation: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>

The default SQLite database is created at `backend/data/app.db`. Database files
are ignored by Git.

## API examples

Submit a run:

```bash
curl -X POST http://127.0.0.1:8000/runs \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello from the TTS system","nfe":8}'
```

Retrieve a run:

```bash
curl http://127.0.0.1:8000/runs/RUN_ID
```

List recent runs:

```bash
curl 'http://127.0.0.1:8000/runs?limit=20&offset=0'
```

Inspect the actual in-process queue without removing jobs:

```bash
curl http://127.0.0.1:8000/queue
```

Example response after a submission:

```json
{"mode":"queue-only","count":1,"run_ids":["the-submitted-run-uuid"]}
```

The count and UUIDs describe waiting queue items in FIFO order. In placeholder
mode, a currently processing item has already left the waiting queue. The
snapshot is observational; it does not claim, dequeue, or process work.

Run statuses are `queued`, `running`, `completed`, and `failed`.

## Frontend connection

By default, CORS allows React development servers at
`http://localhost:5173` and `http://127.0.0.1:5173`. Override the comma-separated
list when necessary:

```bash
export ADAPTIVE_TTS_CORS_ORIGINS='http://localhost:3000,http://localhost:5173'
```

The frontend submits `POST /runs` and can retrieve the accepted request through
`GET /runs/{id}`. With the default queue-only configuration, its state remains
`queued` and there is no result or audio to play.

## Explicit placeholder mode

Legacy processor tests can opt in to the placeholder consumer:

```bash
ADAPTIVE_TTS_WORKER_MODE=placeholder python -m uvicorn backend.app.main:app
```

This mode drains queued work, transitions it through `running`, and stores a
clearly labeled `local-placeholder` result. It does not synthesize speech. Use
an isolated database when trying it; it processes persisted queued requests as
well as new ones. The default does not start this consumer.

## Tests

From the repository root:

```bash
python -m pytest backend/tests -v
```

Tests use a temporary SQLite database and do not modify the development
database.

## Configuration

- `ADAPTIVE_TTS_DATABASE_URL`: overrides the default SQLite URL.
- `ADAPTIVE_TTS_CORS_ORIGINS`: comma-separated allowed frontend origins.
- `ADAPTIVE_TTS_WORKER_MODE`: `queue-only` (default) or explicit `placeholder`.
- `ADAPTIVE_TTS_PROCESSING_DELAY_SECONDS`: placeholder processing delay only.

## Known MVP limitation

The queue lives inside one FastAPI process; run one Uvicorn worker. It is not a
cross-process GPU queue or a Redis service. SQLite preserves queued requests for
recovery after restart, while each process has its own volatile `queue.Queue`.
There is no GPU consumer or remote worker-claim endpoint yet. A future processor
can replace the explicit placeholder path while retaining the run API and state
transitions.


## Real generation consumer

Set `ADAPTIVE_TTS_WORKER_MODE=gpu` in an environment containing the monorepo's patched dots.tts, adaptive-nfe, gpu-workers and wav-validation packages. The queue loop loads `GpuProcessor` once, invokes `GpuWorker.run_job` with the persisted UUID/text/NFE, and saves its actual result and audio path. Exceptions/failed validation become failed runs; the consumer continues.

Completed responses include `audio_url`. `GET /runs/{id}/audio` serves only completed WAVs beneath `ADAPTIVE_TTS_AUDIO_DIR` (default `backend/data/audio`), with byte ranges supported. UTC timestamps serialize with their timezone for browser history.

See [generation setup and Mac CPU verification](../docs/generation-pipeline.md). `bash scripts/dev_mac.sh` from the repo root selects the real float32 CPU environment on Apple Silicon; it does not use the placeholder processor.

New GPU-mode runs attach generation-only telemetry from `gpu-telemetry` and per-run WER/CER from `quality-evals` before completion. Validated WAVs survive optional scorer failures, with errors stored in the result. Failed WAV validation persists its flags and measurements with a failed run. CPU diagnostics work; NVIDIA readings require CUDA/NVML. See [per-run diagnostics and dependency setup](../docs/generation-pipeline.md#per-run-diagnostics).
