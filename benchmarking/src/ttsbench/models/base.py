"""The model adapter contract.

An adapter wraps one TTS implementation. It does *not* time itself: the worker times the
`synthesize` call. Adapters may attribute time to stages with `timer.stage(...)`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from ttsbench.measure.timing import StageTimer


@dataclass(frozen=True)
class Requirements:
    """What a remote backend must install to run this adapter."""

    python: str = "3.11"
    pip: tuple[str, ...] = ()
    apt: tuple[str, ...] = ()


@dataclass(frozen=True)
class SynthesisRequest:
    text: str
    nfe: int
    seed: int
    reference_audio: Path | None = None
    reference_text: str | None = None


@dataclass(frozen=True)
class SynthesisOutput:
    """`audio` may stay on device (e.g. a CUDA tensor); the worker copies it to host untimed."""

    audio: Any
    sample_rate: int
    extras: dict[str, Any] = field(default_factory=dict)


class TTSModel(ABC):
    requirements: ClassVar[Requirements] = Requirements()
    device: str = "cuda"

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def synthesize(self, request: SynthesisRequest, timer: StageTimer) -> SynthesisOutput: ...

    def info(self) -> dict[str, Any]:
        """Static facts stored with the run (checkpoint, dtype, ...)."""
        return {}
