# Adaptive Compute frontend

This is the current text + NFE interface, now part of the main Adaptive Compute repository. The existing logo, palette, fonts, text field, NFE 4/20/32 choices, and arrow button are retained.

## Local development

Use the repository's combined launcher, which starts this frontend on port 3005, the real FastAPI backend on port 8001, and connects them:

```sh
# From the repository root, after installing dependencies described in ../README.md
python3 scripts/dev.py
```

Open http://127.0.0.1:3005/playground. For a separate frontend process:

```sh
# From frontend/
FASTAPI_BASE_URL=http://127.0.0.1:8001 npm run dev -- --hostname 127.0.0.1 --port 3005
```

`.env.example` contains the local setting. Copy it to `.env.local` if you want Next.js to read that setting automatically. The combined launcher supplies it directly and supports alternate ports.

## Requests and receipts

The arrow sends `POST /api/runs` to Next.js with `{ "prompt": "Your text", "nfe": 20 }`. Next.js forwards this to `POST <FASTAPI_BASE_URL>/runs`. The backend validates text (nonblank, at most 5,000 characters), commits a real SQLite row with status `queued`, and puts its UUID in the backend's Python `queue.Queue`.

The backend responds with HTTP 202 and an `id` field. The UI displays that run ID and a **View saved run** link. The link calls `GET /api/runs?id=UUID`, which reads `GET /runs/UUID` from FastAPI and shows the persisted run. `GET /api/runs` lists recent runs. Nothing is seeded; a request is created only by submitting text.

The frontend has no simulated completion, archived samples, or fake generation. In local queue-only mode the request remains queued for GPU worker integration. The Your runs modal sidebar shows a table with text/run ID, NFE, current progress, creation time and audio storage state. Select a row to open its audio, performance, telemetry, quality and full saved result. Active runs refresh by polling every two seconds; there is no SSE endpoint or step/percentage progress yet. Full run metadata persists in localStorage and completed WAV blobs in IndexedDB, separately per browser/origin. Playback uses the saved blob and works after reload with the backend offline. Clearing site data removes local copies, and browser quota errors are reported; Download WAV and Export run details create independent file copies. Removing a run deletes only its local copies, not the server row or WAV. The backend must run in gpu mode to consume queued requests; use bash scripts/dev_mac.sh on this Mac.

`FASTAPI_BASE_URL` stays on the Next.js server, so no browser CORS changes are needed for this flow. Without a configured backend, the UI shows a service-not-connected error. Network failures, timeouts, and backend validation failures preserve the entered text.

## Checks

```sh
npm test
npm run lint
npm run build
```

After building, run the repository-level real-server integration tests from the root:

```sh
.venv/bin/python -m pytest tests/test_local_integration.py -v
```

All request tests use isolated test services/databases; the development SQLite database is left untouched.

Run details include dedicated WAV validation, telemetry and quality sections. New real-worker runs collect the team’s generation telemetry and score accepted clips with Whisper WER/CER. CPU runs retain timing/runtime metadata and label GPU-only measurements unavailable. Optional speaker similarity requires a reference recording; scorer errors preserve playable audio. Older runs are not backfilled with invented telemetry or scores. See [diagnostics setup](../docs/generation-pipeline.md#per-run-diagnostics).
