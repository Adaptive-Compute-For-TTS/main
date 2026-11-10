"""Worker settings, fixed for the worker's lifetime (the model is loaded once with them).

Every setting can come from an environment variable (GPU_WORKER_<NAME>), so the same code runs on
any GPU machine without edits. Defaults match gpu-executor/scripts/generate.py and the ml-evals
baseline (seed 42, guidance 1.2, speaker scale 1.5, bf16).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Mapping


def _bool(value: str) -> bool:
    v = value.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(f"expected a boolean, got {value!r}")


def _optional_path(value: str) -> Path | None:
    return Path(value) if value.strip() else None


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(x) for x in value.replace(",", " ").split())


@dataclass(frozen=True)
class WorkerConfig:
    model: str = "dots-studio/dots.tts-soar"
    precision: str = "bfloat16"
    # False: eager path. True: dots.tts's fast path (torch.compile + KV cache), ~3 min compile at start.
    optimize: bool = False
    seed: int = 42
    guidance_scale: float = 1.2
    speaker_scale: float = 1.5
    # Where wav-validation stores clips: <storage_dir>/<run_id>/audio.wav, rejected under _rejected/.
    # Still to agree with the backend (Sai); runs/storage is gitignored.
    storage_dir: Path = Path("runs/storage")
    # Discarded runs at startup, one per NFE, so the first real run doesn't pay for setup.
    warmup_text: str = "Hello, this is a warmup run."
    warmup_nfes: tuple[int, ...] = field(default=(10,))
    # The voice quality eval compares each clip against (speaker similarity, `sim_ref`). The worker
    # generates without a prompt voice, so None means sim_ref stays None; WER/CER still work.
    reference_wav: Path | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "WorkerConfig":
        """Defaults, overridden by GPU_WORKER_MODEL, GPU_WORKER_OPTIMIZE, GPU_WORKER_STORAGE_DIR, ...

        GPU_WORKER_REFERENCE_WAV is the reference voice for speaker similarity (empty or unset: none).
        GPU_WORKER_WARMUP_NFES takes a list ("4 10 32" or "4,10,32"); empty means no warmup.
        """
        env = os.environ if env is None else env
        parse = {"optimize": _bool, "seed": int, "guidance_scale": float, "speaker_scale": float,
                 "storage_dir": Path, "warmup_nfes": _ints, "reference_wav": _optional_path}
        kwargs = {}
        for f in fields(cls):
            key = f"GPU_WORKER_{f.name.upper()}"
            if key in env:
                kwargs[f.name] = parse.get(f.name, str)(env[key])
        return cls(**kwargs)
