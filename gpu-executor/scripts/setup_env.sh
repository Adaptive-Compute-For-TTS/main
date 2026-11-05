#!/usr/bin/env bash
# Creates gpu-executor/.venv and installs torch + dots.tts + this component's package.
# Run from gpu-executor/:  bash scripts/setup_env.sh        (CUDA build)
#                          bash scripts/setup_env.sh cpu    (CPU build)
# Expects the dots.tts clone (with patches/dots_tts_scheduler_hook.patch applied) next to this
# monorepo, i.e. ../../dots.tts from here. Override with DOTS_TTS_DIR=/path/to/dots.tts.
set -euo pipefail

DOTS_TTS_DIR="${DOTS_TTS_DIR:-../../dots.tts}"

DEVICE="${1:-cuda}"
case "$DEVICE" in
  cuda) INDEX=https://download.pytorch.org/whl/cu128 ;;
  cpu)  INDEX=https://download.pytorch.org/whl/cpu ;;
  *) echo "usage: $0 [cuda|cpu]" >&2; exit 1 ;;
esac

[ -f "$DOTS_TTS_DIR/pyproject.toml" ] || { echo "dots.tts clone not found at $DOTS_TTS_DIR (set DOTS_TTS_DIR)" >&2; exit 1; }
[ -x .venv/bin/python ] || python3 -m venv .venv
PY=.venv/bin/python

$PY -m pip install --upgrade pip

# 1. torch first, from the index matching the hardware, so pip can't pick a different build later.
$PY -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url "$INDEX"

# 2. dots.tts (editable, so the hook patch in its source is what runs), this package (editable), test deps.
#    Pinned by constraints.txt.
$PY -m pip install -e "$DOTS_TTS_DIR" -r requirements.txt -c constraints.txt

# 3. Sanity check.
$PY -c "import torch, dots_tts, adaptive_nfe; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
