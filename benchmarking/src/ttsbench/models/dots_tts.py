"""Adapter for dots.tts (https://github.com/studio-dots-ai/dots.tts).

NFE maps to `num_steps` of the flow-matching sampler. Finer stage attribution (text encoder,
per-step transformer forward, AudioVAE decode) needs hooks inside `DotsTtsRuntime.generate`;
until those exist the whole call is reported as the single stage "generate".
"""

from __future__ import annotations

import sys
from typing import Any

from ttsbench.measure.timing import StageTimer
from ttsbench.models.base import Requirements, SynthesisOutput, SynthesisRequest, TTSModel

# Pinned for reproducible images: tag v0.3.1, plus upstream constraints/recommended.txt with
# gradio bumped 6.17.0 -> 6.17.3 (6.17.0 is no longer on PyPI, so the upstream file can't resolve).
REVISION = "09235d55005c61867d3b94aeb6e13eede919a0a3"
# dots.tts's own dependencies, shared with the dots_tts_executor model.
DEPENDENCY_PINS = (
    "torch==2.8.0",
    "torchaudio==2.8.0",
    "transformers==4.57.0",
    "librosa==0.11.0",
    "soundfile==0.13.1",
    "numpy==2.2.6",
    "pydantic==2.12.5",
    "PyYAML==6.0.3",
    "safetensors==0.8.0rc0",
    "gradio==6.17.3",
)
_PINS = (f"dots.tts @ git+https://github.com/studio-dots-ai/dots.tts.git@{REVISION}", *DEPENDENCY_PINS)


class DotsTTS(TTSModel):
    requirements = Requirements(python="3.11", pip=_PINS, apt=("git", "ffmpeg", "libsndfile1"))
    device = "cuda"

    def __init__(
        self,
        checkpoint: str = "dots-studio/dots.tts-soar",
        precision: str = "bfloat16",
        guidance_scale: float = 1.2,
    ) -> None:
        self.checkpoint = checkpoint
        self.precision = precision
        self.guidance_scale = guidance_scale
        self._runtime: Any = None

    def load(self) -> None:
        from dots_tts.runtime import DotsTtsRuntime
        from loguru import logger

        # dots.tts logs DEBUG lines per audio patch; that I/O would land inside the timed region.
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

        self._runtime = DotsTtsRuntime.from_pretrained(self.checkpoint, precision=self.precision)

    def synthesize(self, request: SynthesisRequest, timer: StageTimer) -> SynthesisOutput:
        kwargs: dict[str, Any] = {
            "text": request.text,
            "num_steps": request.nfe,
            "guidance_scale": self.guidance_scale,
        }
        if request.reference_audio is not None:
            kwargs["prompt_audio_path"] = str(request.reference_audio)
            kwargs["prompt_text"] = request.reference_text
        with timer.stage("generate"):
            result = self._runtime.generate(**kwargs)
        return SynthesisOutput(audio=result["audio"], sample_rate=int(result["sample_rate"]))

    def info(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "dots_tts_revision": REVISION,
            "precision": self.precision,
            "guidance_scale": self.guidance_scale,
        }
