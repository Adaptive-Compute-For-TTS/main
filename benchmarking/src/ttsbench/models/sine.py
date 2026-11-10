"""A deterministic CPU stand-in for a flow-matching TTS model.

Cost scales with NFE and text length, so the whole pipeline (sweeps, storage, summaries,
comparisons) can be exercised without a GPU or model weights.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ttsbench.measure.timing import StageTimer
from ttsbench.models.base import Requirements, SynthesisOutput, SynthesisRequest, TTSModel


class SineModel(TTSModel):
    requirements = Requirements(pip=("numpy",))
    device = "cpu"

    def __init__(self, sample_rate: int = 24_000, seconds_per_char: float = 0.06, work: int = 64) -> None:
        self.sample_rate = sample_rate
        self.seconds_per_char = seconds_per_char
        self.work = work
        self._basis: np.ndarray | None = None

    def load(self) -> None:
        self._basis = np.random.default_rng(0).standard_normal((self.work, self.work)).astype(np.float32)

    def synthesize(self, request: SynthesisRequest, timer: StageTimer) -> SynthesisOutput:
        assert self._basis is not None, "load() was not called"
        rng = np.random.default_rng(request.seed)
        n = max(1, int(len(request.text) * self.seconds_per_char * self.sample_rate))
        frames = max(1, n // 256)

        with timer.stage("encode_text"):
            latent = rng.standard_normal((frames, self.work)).astype(np.float32)

        for _ in range(request.nfe):
            with timer.stage("flow_step"):
                latent = np.tanh(latent @ self._basis * 0.1) + latent

        with timer.stage("vocoder"):
            t = np.arange(n, dtype=np.float32) / self.sample_rate
            pitch = 110.0 + 20.0 * float(np.tanh(latent.mean()))
            audio = 0.3 * np.sin(2 * np.pi * pitch * t, dtype=np.float32)

        return SynthesisOutput(audio=audio, sample_rate=self.sample_rate)

    def info(self) -> dict[str, Any]:
        return {"sample_rate": self.sample_rate, "work": self.work}
