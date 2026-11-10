"""dots.tts through the team's fixed-NFE executor (`gpu-executor/`, package `adaptive_nfe`).

The image installs dots.tts at the commit `gpu-executor` patches, with
`gpu-executor/patches/dots_tts_scheduler_hook.patch` applied, plus `adaptive_nfe` from this repo.
Every mode runs on that same build, so runs of different modes compare directly:

- `stock`: plain `runtime.generate(num_steps=nfe)`. The hooks stay None, so this is stock dots.tts.
- `scheduler`: `generate_fixed_nfe(use_executor=False)`. The scheduler freezes the NFE, and
  dots.tts's own DiT loop runs it.
- `executor`: `generate_fixed_nfe`. Our executor runs every patch's DiT loop (`decode_patch`, or
  `decode_patch_optimized` with `optimize: true`).

`optimize` is dots.tts's fast path (torch.compile + CUDA graphs). It compiles on the first call
per shape, which lands in the worker's per-NFE warmup, so set `measurement.warmup` >= 1. Warmup
runs on the first prompt only, so a longer prompt that needs a bigger KV-cache bucket can still pay
a reallocation + prefill in a measured sample (same for stock dots.tts); check p99 vs p50.
`generate_fixed_nfe` checks the per-patch NFE log after each call. That check is inside the timed
region, but it is a list comparison and costs microseconds.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ttsbench.measure.timing import StageTimer
from ttsbench.models.base import Requirements, SynthesisOutput, SynthesisRequest, TTSModel
from ttsbench.models.dots_tts import DEPENDENCY_PINS


def _find_gpu_executor() -> Path:
    """The monorepo's gpu-executor/, searched upward from this file. Only the client needs it (to
    build the image); in a remote worker ttsbench sits at e.g. /root/ttsbench, where it isn't found
    and the placeholder path is never read."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "gpu-executor" / "patches").is_dir():
            return parent / "gpu-executor"
    return Path("gpu-executor")


GPU_EXECUTOR = _find_gpu_executor()
HOOK_PATCH = "gpu-executor/patches/dots_tts_scheduler_hook.patch"
# The dots.tts commit the hook patch applies to (gpu-executor/README.md, "Setup"). v0.3.1 + 3 commits.
DOTS_REVISION = "3cb6f94a571097b500038f1619959075282c5c98"
MODES = ("stock", "scheduler", "executor")

_SETUP = (
    'rm -rf "{build}/dots.tts" && git clone -q https://github.com/studio-dots-ai/dots.tts.git "{build}/dots.tts"',
    f'git -C "{{build}}/dots.tts" checkout -q {DOTS_REVISION}',
    f'git -C "{{build}}/dots.tts" apply "{{build}}/{HOOK_PATCH}"',
    '{pip} "{build}/dots.tts" "{build}/gpu-executor"',
)


class DotsTTSExecutor(TTSModel):
    requirements = Requirements(
        python="3.11",
        pip=DEPENDENCY_PINS,
        apt=("git", "ffmpeg", "libsndfile1"),
        sources=(
            (GPU_EXECUTOR / "pyproject.toml", "gpu-executor/pyproject.toml"),
            (GPU_EXECUTOR / "src", "gpu-executor/src"),
            (GPU_EXECUTOR / "patches", "gpu-executor/patches"),
        ),
        setup=_SETUP,
    )
    device = "cuda"

    def __init__(
        self,
        mode: str = "executor",
        optimize: bool = False,
        checkpoint: str = "dots-studio/dots.tts-soar",
        precision: str = "bfloat16",
        guidance_scale: float = 1.2,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}; got {mode!r}")
        self.mode = mode
        self.optimize = optimize
        self.checkpoint = checkpoint
        self.precision = precision
        self.guidance_scale = guidance_scale
        self._runtime: Any = None
        self._executor: Any = None

    def load(self) -> None:
        from dots_tts.runtime import DotsTtsRuntime
        from loguru import logger

        # dots.tts logs DEBUG lines per audio patch; that I/O would land inside the timed region.
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

        self._runtime = DotsTtsRuntime.from_pretrained(
            self.checkpoint, precision=self.precision, optimize=self.optimize
        )
        if self.mode == "executor":
            from adaptive_nfe.executor import Executor

            self._executor = Executor(self._runtime.model)  # built once, reused by every call

    def synthesize(self, request: SynthesisRequest, timer: StageTimer) -> SynthesisOutput:
        kwargs: dict[str, Any] = {"text": request.text, "guidance_scale": self.guidance_scale}
        if request.reference_audio is not None:
            kwargs["prompt_audio_path"] = str(request.reference_audio)
            kwargs["prompt_text"] = request.reference_text

        if self.mode == "stock":
            with timer.stage("generate"):
                output = self._runtime.generate(num_steps=request.nfe, **kwargs)
            return SynthesisOutput(audio=output["audio"], sample_rate=int(output["sample_rate"]))

        from adaptive_nfe import generate_fixed_nfe

        with timer.stage("generate"):
            result = generate_fixed_nfe(
                self._runtime,
                request.nfe,
                use_executor=self.mode == "executor",
                executor=self._executor,
                **kwargs,
            )
        extras = {"num_patches": result.num_patches, "nfe_per_patch": list(result.nfe_per_patch)}
        return SynthesisOutput(
            audio=result.output["audio"], sample_rate=int(result.output["sample_rate"]), extras=extras
        )

    def info(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "optimize": self.optimize,
            "checkpoint": self.checkpoint,
            "dots_tts_revision": DOTS_REVISION,
            "hook_patch": HOOK_PATCH,
            "precision": self.precision,
            "guidance_scale": self.guidance_scale,
        }
