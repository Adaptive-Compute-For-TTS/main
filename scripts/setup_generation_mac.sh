#!/usr/bin/env bash
# Native Apple Silicon CPU correctness environment; no model download or database writes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo 'This setup targets native Apple Silicon macOS.' >&2; exit 1; }
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }
mkdir -p .local
if [[ ! -x .local/bin/micromamba ]]; then
  curl -fL https://micro.mamba.pm/api/micromamba/osx-arm64/latest -o .local/micromamba.tar.bz2
  tar -xjf .local/micromamba.tar.bz2 -C .local bin/micromamba
fi
DOTS_REVISION=32407a55228630475c48ecdb2c4e2c0f9c09e030
if [[ ! -d .local/dots.tts/.git ]]; then
  git clone https://github.com/studio-dots-ai/dots.tts.git .local/dots.tts
  git -C .local/dots.tts checkout --detach "$DOTS_REVISION"
fi
[[ "$(git -C .local/dots.tts rev-parse HEAD)" == "$DOTS_REVISION" ]] || { echo 'Existing local dots.tts clone has a different revision; preserve it and inspect before changing it.' >&2; exit 1; }
PATCH="$ROOT/gpu-executor/patches/dots_tts_scheduler_hook.patch"
if git -C .local/dots.tts apply --check "$PATCH" 2>/dev/null; then
  git -C .local/dots.tts apply "$PATCH"
elif ! git -C .local/dots.tts apply --reverse --check "$PATCH" 2>/dev/null; then
  echo 'The fixed-NFE patch is neither applicable nor already installed; inspect the clone.' >&2
  exit 1
fi
MAMBA_ROOT_PREFIX="$ROOT/.local/mamba" .local/bin/micromamba create -y \
  -p "$ROOT/.local/tts-env" -c conda-forge --override-channels python=3.12 pynini=2.1.7 pip
uv pip install --python .local/tts-env/bin/python -r backend/requirements.txt \
  -r quality-evals/speaker_sim/requirements-worker.txt nvidia-ml-py \
  -e .local/dots.tts -e gpu-executor -e gpu-workers -e wav-validation \
  -c gpu-executor/constraints.txt
.local/tts-env/bin/python -c 'import platform,torch,gpu_worker,dots_tts,pynini; print(platform.machine(),torch.__version__,"CUDA:",torch.cuda.is_available())'
echo 'Ready. CPU float32 generation downloads the model on the first actual request.'
echo "GPU_WORKER_PRECISION=float32 GPU_WORKER_OPTIMIZE=false GPU_WORKER_WARMUP_NFES='' python3 scripts/dev.py --worker-mode gpu --python .local/tts-env/bin/python"
