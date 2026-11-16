# Adaptive Compute

The local request path is:

**Frontend arrow → Next.js `POST /api/runs` → FastAPI `POST /runs` → SQLite run record + Python `queue.Queue`.**

The frontend lives in `frontend/` and the FastAPI service lives in `backend/`. The default local mode leaves requests `queued` for a GPU worker. It does not generate placeholder results or pretend speech was completed.

## Install once

From this repository's root, using Python 3.9+ and Node.js 22+:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.txt
npm --prefix frontend ci
```

## Run real speech generation on this Mac

```sh
bash scripts/setup_generation_mac.sh  # one-time native CPU dependencies
bash scripts/dev_mac.sh
```

This runs real dots.tts with the fixed-NFE executor on Apple Silicon CPU in float32. First use downloads the model. The browser's **Your runs** panel shows progress and plays the validated output. For a CUDA machine, run `python3 scripts/dev.py --worker-mode gpu --python /path/to/gpu-env/bin/python` after installing the GPU environment.

## Run submission-only inspection locally

```sh
python3 scripts/dev.py
```

This starts both services and supplies the backend URL to the frontend:

- Frontend: [http://127.0.0.1:3005/playground](http://127.0.0.1:3005/playground)
- FastAPI documentation: [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)
- Actual in-memory queue: [http://127.0.0.1:8001/queue](http://127.0.0.1:8001/queue)
- Persisted runs: [http://127.0.0.1:8001/runs](http://127.0.0.1:8001/runs)

Enter text, choose NFE 4, 20, or 32, and click the arrow. The server creates a run with an `id`, saves it to SQLite, and puts the same ID in the queue. Starting the services creates the database schema if needed; it does not seed or submit any runs. Existing user runs are preserved. Press Ctrl+C to stop both services.

Port 8001 avoids another application already using port 8000 on this device. To choose different ports:

```sh
python3 scripts/dev.py --backend-port 8011 --frontend-port 3015
```

The launcher refuses occupied ports instead of connecting to an unrelated service. It runs one FastAPI process. Restart it after backend source changes; the frontend reloads its own changes automatically.

## Inspect your request

For a readable, read-only queue and recent-runs view:

```sh
python3 scripts/inspect_queue.py
python3 scripts/inspect_queue.py --watch
```

For custom ports, pass `--backend-url http://127.0.0.1:8011`.

The inspector displays actual queue membership, run IDs, status, NFE, and text. It does not consume jobs. `GET /runs/ID` retrieves a specific persisted run. The default SQLite file is `backend/data/app.db`, ignored by Git. To inspect it directly without changing it:

```sh
sqlite3 -readonly backend/data/app.db \
  'SELECT id, prompt, nfe, status, created_at FROM runs ORDER BY created_at DESC;'
```

The queue is a Python `queue.Queue` inside the FastAPI process, not a cross-process broker. A GPU worker consuming this exact queue must run in that process; a separately launched Python process cannot share it by importing the module. Queued SQLite rows are recovered into the queue when FastAPI restarts. In queue-only inspection mode no consumer runs. In GPU mode the existing worker/executor consumes jobs, validates WAVs, persists results and serves completed audio; browser history polls these statuses.

## Test

```sh
.venv/bin/python -m pytest backend/tests -v
npm --prefix frontend test
npm --prefix frontend run lint
npm --prefix frontend run build
.venv/bin/python -m pytest tests/test_local_integration.py -v
```

The integration test starts the real FastAPI and built Next.js servers on free ports, submits through the frontend proxy, and checks SQLite and the actual in-memory queue. Build the frontend before running it. Automated request tests use isolated temporary SQLite databases. They do not seed or modify `backend/data/app.db`.

## Run services separately

From the repository root:

```sh
ADAPTIVE_TTS_WORKER_MODE=queue-only \
  .venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8001
```

In another terminal:

```sh
cd frontend
FASTAPI_BASE_URL=http://127.0.0.1:8001 npm run dev -- --hostname 127.0.0.1 --port 3005
```

`FASTAPI_BASE_URL` is a server-side setting. The browser sends to the frontend's same-origin `/api/runs` proxy, so this path does not need browser CORS configuration. To use another SQLite database, set `ADAPTIVE_TTS_DATABASE_URL` before starting FastAPI. The optional placeholder worker requires explicit `ADAPTIVE_TTS_WORKER_MODE=placeholder`; the combined launcher always uses queue-only mode.

This workflow runs locally. A hosted frontend needs a backend address reachable from its hosting service; the existing Vercel deployment cannot reach a server bound to this Mac's loopback interface.


## Speech generation and browser history

The corner **Your runs** panel remembers this browser's submitted runs across refresh, polls active jobs and shows an audio player on completion. Generation uses the existing gpu-workers → fixed-NFE executor → WAV validator/storage code, consumed by the same in-process queue as FastAPI. The secure audio endpoint and Next.js proxy support seeking.

To enable the real consumer, use `scripts/dev.py --worker-mode gpu --python /path/to/worker-env/bin/python`; queue-only mode remains the default for inspection. On this Apple Silicon Mac, the CPU correctness environment is `.local/tts-env` (native Python 3.12, float32). Run `bash scripts/setup_generation_mac.sh` to provision it on a fresh checkout.

See [the generation architecture and Mac test procedure](docs/generation-pipeline.md). CPU correctness tests and real-model generation are distinct from CUDA performance measurements. No fabricated audio processor is available through the normal launcher.
