#!/usr/bin/env bash
# Sets up the worker's environment: gpu-executor's venv (torch + dots.tts + adaptive-nfe) plus
# wav-validation and this package, all editable. One venv, gpu-executor/.venv, shared on purpose:
# the worker runs exactly the executor code that gpu-executor's checks test.
#
# Run from anywhere:  bash gpu-workers/scripts/setup_env.sh        (CUDA build)
#                     bash gpu-workers/scripts/setup_env.sh cpu    (CPU build, tests only)
# Needs the dots.tts clone (hook patch applied) next to the monorepo; see gpu-executor/README.md.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"   # gpu-workers/
ROOT="$(dirname "$HERE")"                  # monorepo root

(cd "$ROOT/gpu-executor" && bash scripts/setup_env.sh "${1:-cuda}")
PY="$ROOT/gpu-executor/.venv/bin/python"
$PY -m pip install -e "$ROOT/wav-validation" -e "$HERE[test]"
$PY -c "import gpu_worker, wav_validation, adaptive_nfe; print('gpu_worker', gpu_worker.__version__, '| wav_validation', wav_validation.__version__)"
echo "Activate with: source $ROOT/gpu-executor/.venv/bin/activate"
