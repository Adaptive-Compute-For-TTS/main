"""Data contracts shared by the client, the backends and the worker.

Everything that crosses a process/network boundary (client -> backend -> worker) is a pydantic
model so it is validated on both sides and serialises to plain dicts.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Bump whenever the way latency is measured changes (sync points, what is inside the timed
# region, warmup policy, ...). Stored with every run so numbers from different protocols are
# never compared silently.
MEASUREMENT_PROTOCOL_VERSION = 1

LengthBucket = Literal["short", "medium", "long"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------


class Prompt(_Frozen):
    """One benchmark input. `reference_audio` is relative to the assets root (benchmarks/)."""

    id: str
    text: str
    length: LengthBucket
    language: str = "en"
    tags: tuple[str, ...] = ()
    reference_audio: str | None = None
    reference_text: str | None = None

    @field_validator("id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
        if not value or not all(c.isalnum() or c in "-_." for c in value):
            raise ValueError(f"prompt id must be non-empty and filesystem-safe: {value!r}")
        return value


class ComponentSpec(_Frozen):
    """A registry name plus constructor kwargs (used for models and backends)."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class MeasurementConfig(_Frozen):
    warmup: int = Field(default=3, ge=0, description="Discarded runs per NFE value before measuring.")
    repeats: int = Field(default=5, ge=1, description="Measured runs per (prompt, NFE).")
    seed: int = 0
    vary_seed: bool = Field(
        default=False,
        description="If false every repeat uses the same seed, so audio should be bit-identical.",
    )
    save_audio: Literal["all", "first", "none"] = "first"
    disable_gc: bool = True


class SweepConfig(_Frozen):
    nfe: tuple[int, ...] = Field(min_length=1)

    @field_validator("nfe")
    @classmethod
    def _positive(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(n < 1 for n in value):
            raise ValueError("nfe values must be >= 1")
        if len(set(value)) != len(value):
            raise ValueError("nfe values must be unique")
        return value


class PromptSelection(_Frozen):
    path: str = "benchmarks/prompts/v0.jsonl"
    lengths: tuple[LengthBucket, ...] = ()
    tags: tuple[str, ...] = ()
    ids: tuple[str, ...] = ()
    limit: int | None = Field(default=None, ge=1)


class ExperimentConfig(_Frozen):
    name: str
    description: str = ""
    model: ComponentSpec
    backend: ComponentSpec = ComponentSpec(name="local")
    prompts: PromptSelection = PromptSelection()
    sweep: SweepConfig
    measurement: MeasurementConfig = MeasurementConfig()
    probes: tuple[str, ...] = ()
    output_dir: str = "runs"


# --------------------------------------------------------------------------------------------
# Worker job contract
# --------------------------------------------------------------------------------------------


class Job(_Frozen):
    """A unit of work executed inside one worker process: all prompts for one NFE value."""

    model: ComponentSpec
    nfe: int
    prompts: tuple[Prompt, ...]
    measurement: MeasurementConfig
    probes: tuple[str, ...] = ()


class Sample(_Frozen):
    """One measured synthesis. All times in milliseconds."""

    prompt_id: str
    nfe: int
    repeat: int
    seed: int
    latency_ms: float = Field(description="Host wall time, device-synchronised on both ends.")
    device_ms: float | None = Field(default=None, description="CUDA-event time on the device stream.")
    to_host_ms: float = Field(description="Time to copy the output audio to host memory (not in latency).")
    stages_ms: dict[str, float] = Field(default_factory=dict)
    stage_calls: dict[str, int] = Field(default_factory=dict)
    audio_seconds: float
    sample_rate: int
    rtf: float = Field(description="latency / audio duration; < 1 is faster than real time.")
    audio_sha256: str
    audio_path: str | None = None
    telemetry: dict[str, Any] = Field(default_factory=dict)
    extras: dict[str, Any] = Field(default_factory=dict, description="Model-reported facts, e.g. NFE per patch.")


class WorkerInfo(_Frozen):
    environment: dict[str, Any]
    model_info: dict[str, Any]
    model_load_ms: float | None = Field(description="Set only on the call that loaded the model.")
    cold_start: bool
    warmup_ms: tuple[float, ...]


class JobResult(BaseModel):
    """Not frozen: carries raw audio bytes (float32 WAV) keyed by `audio_key()`.

    Bytes are base64 in JSON so batch backends can move results as plain files.
    """

    model_config = ConfigDict(extra="forbid", ser_json_bytes="base64", val_json_bytes="base64")

    nfe: int
    samples: list[Sample]
    audio: dict[str, bytes]
    worker: WorkerInfo


def audio_key(prompt_id: str, nfe: int, repeat: int) -> str:
    return f"{prompt_id}/nfe{nfe:03d}_r{repeat:02d}"
