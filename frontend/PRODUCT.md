# Product

<!-- impeccable:product-schema 1 -->

## Platform
web

## Users
The Adaptive Compute team and people using its text and compute input interface.

## Product Purpose
A single-page interface for entering freeform text, choosing a fixed NFE budget of 4, 20, or 32, and submitting the request to FastAPI.

## Capabilities and Constraints
The page has an unrestricted text area and one native radio group for the three NFE choices. NFE 20 is selected initially. The header retains the original logo, uses the name Adaptive Compute at the top left, and leaves the top right empty.

Text and selection are held in page memory. An arrow button submits nonblank text and the selected NFE through POST /api/runs. The server forwards {prompt, nfe} to POST /runs on the service configured by FASTAPI_BASE_URL. The local launcher supplies FASTAPI_BASE_URL for the real backend on port 8001. Refreshing starts with empty text and NFE 20.

During submission, inputs and the submit button are disabled to prevent duplicate requests. Status and error messages are announced accessibly; input is preserved for retries. Success displays the backend id and a link to the saved SQLite run. It confirms durable queue submission, not completed speech generation. A corner Your runs button opens a modal sidebar with a semantic runs table (text/ID, NFE, current progress, creation time, and local audio state). Selecting a run opens its audio, full performance, WAV validation (including cautions), telemetry, quality and future metrics. New FastAPI worker runs collect generation-only CPU/GPU telemetry and evaluate validated audio using the team’s Whisper WER/CER scorer. NVIDIA-only fields are marked unavailable on CPU. Speaker similarity requires a configured reference recording; errors and skipped measurements are explicit. Historical records retain their actual available data; the UI does not invent missing metrics. Active runs poll every two seconds; SSE and percentage/step progress are not implemented. Full run metadata persists in localStorage, and completed WAV blobs are committed to IndexedDB before playback. Cached audio and metrics remain available with the backend offline. Browser storage is per origin, survives reloads, and can be removed by clearing site data; persistent storage is requested on opening the panel. Download WAV and Export run details provide file copies. Removing a local run removes its browser metadata and WAV only. There is no arbitrary history count cap; quota errors are reported. The interface has no benchmark catalog, recommendations, or API-key setup.

## Visual Identity
Retain the existing warm stone and ink palette, Geist typography, thin borders, and restrained spacing. The text area and NFE choices appear together on desktop and stack on narrow screens. Labels and keyboard focus remain visible.

## Existing Hosting
The existing Vercel project is `project-58c2/watai-nfe-playground`, at [the playground](https://watai-nfe-playground.vercel.app/playground). Source changes require a deployment to update that hosted version.
