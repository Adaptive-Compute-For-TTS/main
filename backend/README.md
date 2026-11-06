# Adaptive Compute TTS Backend

Local FastAPI backend for submitting TTS runs, storing their state in SQLite,
and exposing results to the React frontend.

The current worker is intentionally a placeholder. It proves the complete
request-to-result flow and provides the integration boundary for the future GPU
worker.

## Current flow

1. `POST /runs` validates a prompt and an NFE value from 4 through 32.
2. The backend creates a SQLite record with status `queued`.
3. A local Python worker changes the status to `running`.
4. The placeholder processor stores a result and marks the run `completed`.
5. The frontend can poll `GET /runs/{id}` for the latest state.

Failures are persisted with status `failed` and an `error_message`.

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

Run statuses are `queued`, `running`, `completed`, and `failed`.

## Frontend connection

By default, CORS allows React development servers at
`http://localhost:5173` and `http://127.0.0.1:5173`. Override the comma-separated
list when necessary:

```bash
export ADAPTIVE_TTS_CORS_ORIGINS='http://localhost:3000,http://localhost:5173'
```

The first frontend integration should poll `GET /runs/{id}` every one or two
seconds until the status is `completed` or `failed`.

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
- `ADAPTIVE_TTS_PROCESSING_DELAY_SECONDS`: placeholder processing delay.

## Known MVP limitation

The worker queue lives inside the FastAPI process. Pending in-memory queue items
do not survive a server restart, and the app should run with one Uvicorn worker.
The API and repository contracts are designed so a persistent GPU worker can
replace `build_placeholder_result` later without changing the frontend API.
