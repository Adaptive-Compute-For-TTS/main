"""The model adapter contract.

An adapter wraps one TTS implementation. It does *not* time itself: the worker times the
`synthesize` call. Adapters may attribute time to stages with `timer.stage(...)`.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from ttsbench.measure.timing import StageTimer


def source_files(local: Path) -> list[Path]:
    """The files a source path ships: itself, or every file under it except caches."""
    if not local.is_dir():
        return [local]
    return sorted(p for p in local.rglob("*") if p.is_file() and "__pycache__" not in p.parts)


@dataclass(frozen=True)
class Requirements:
    """What a remote backend must install to run this adapter.

    Install order: `apt`, `pip`, then `sources` are copied and `setup` runs. Each source is a
    `(local_path, name)` pair, copied to `<build dir>/<name>`. `setup` holds shell commands where
    `{build}` is the build dir and `{pip}` installs into the run's Python environment.
    """

    python: str = "3.11"
    pip: tuple[str, ...] = ()
    apt: tuple[str, ...] = ()
    sources: tuple[tuple[Path, str], ...] = ()
    setup: tuple[str, ...] = ()

    def setup_commands(self, build: str, pip: str) -> list[str]:
        return [command.format(build=build, pip=pip) for command in self.setup]

    def sources_digest(self) -> str:
        """Hash of every source file's name and bytes, so editing a source rebuilds the env."""
        digest = hashlib.sha256()
        for local, name in self.sources:
            for path in source_files(local):
                rel = path.relative_to(local).as_posix() if local.is_dir() else ""
                digest.update(f"{name}/{rel}\0".encode())
                digest.update(path.read_bytes())
        return digest.hexdigest()


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
