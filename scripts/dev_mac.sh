#!/usr/bin/env bash
# Real dots.tts generation on Apple Silicon CPU, not a synthetic-audio demo.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[[ -x .local/tts-env/bin/python ]] || { echo 'Run bash scripts/setup_generation_mac.sh first.' >&2; exit 1; }
export GPU_WORKER_PRECISION=float32
export GPU_WORKER_OPTIMIZE=false
export GPU_WORKER_WARMUP_NFES=''
exec .local/tts-env/bin/python scripts/dev.py --worker-mode gpu --python "$ROOT/.local/tts-env/bin/python" "$@"
