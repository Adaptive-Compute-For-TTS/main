"""The measured loop. Runs *inside* the execution environment (local process or remote GPU
container), so transport latency to a backend is never part of a measurement.

Backends only have to get a `Job` here and a `JobResult` back.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

from ttsbench.audio import encode_wav, fingerprint, to_numpy
from ttsbench.measure.environment import worker_environment
from ttsbench.measure.probes import PROBES, Probe
from ttsbench.measure.timing import Device, host_timed, measure
from ttsbench.models import MODELS, SynthesisRequest, TTSModel
from ttsbench.schema import ComponentSpec, Job, JobResult, Prompt, Sample, WorkerInfo, audio_key

ASSETS_ROOT_ENV = "TTSBENCH_ASSETS_ROOT"

# One loaded model per process, keyed by its spec, so a warm container reuses it across jobs.
_LOADED: dict[str, tuple[TTSModel, Device]] = {}


def _spec_key(spec: ComponentSpec) -> str:
    return json.dumps(spec.model_dump(), sort_keys=True)


def _load(spec: ComponentSpec) -> tuple[TTSModel, Device, float | None]:
    key = _spec_key(spec)
    if key in _LOADED:
        model, device = _LOADED[key]
        return model, device, None
    model = MODELS.create(spec.name, **spec.params)
    device = Device(model.device)
    _, load_ms = host_timed(model.load)
    _LOADED.clear()  # never keep two models resident on the GPU
    _LOADED[key] = (model, device)
    return model, device, load_ms


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _request(prompt: Prompt, nfe: int, seed: int, assets_root: Path) -> SynthesisRequest:
    reference = assets_root / prompt.reference_audio if prompt.reference_audio else None
    if reference is not None and not reference.is_file():
        raise FileNotFoundError(f"reference audio for prompt {prompt.id!r} not found: {reference}")
    return SynthesisRequest(
        text=prompt.text,
        nfe=nfe,
        seed=seed,
        reference_audio=reference,
        reference_text=prompt.reference_text,
    )


def _should_keep_audio(policy: str, repeat: int) -> bool:
    return policy == "all" or (policy == "first" and repeat == 0)


def execute_job(job: Job, assets_root: Path | None = None) -> JobResult:
    assets_root = assets_root or Path(os.environ.get(ASSETS_ROOT_ENV, "benchmarks"))
    model, device, load_ms = _load(job.model)
    probes: dict[str, Probe] = {name: PROBES.create(name, device=device) for name in job.probes}
    m = job.measurement

    def synth(request: SynthesisRequest) -> Any:
        return lambda timer: model.synthesize(request, timer)

    # Warmup: discarded, on the first prompt, at this job's NFE (captures compile/graph/autotune
    # cost per NFE shape so it never lands in measured samples).
    warmup_ms = []
    for i in range(m.warmup):
        seed_everything(m.seed + i)
        request = _request(job.prompts[0], job.nfe, m.seed + i, assets_root)
        _, timing = measure(synth(request), device, disable_gc=m.disable_gc)
        warmup_ms.append(timing.latency_ms)

    samples: list[Sample] = []
    audio: dict[str, bytes] = {}
    for prompt in job.prompts:
        for repeat in range(m.repeats):
            seed = m.seed + repeat if m.vary_seed else m.seed
            request = _request(prompt, job.nfe, seed, assets_root)
            seed_everything(seed)
            for probe in probes.values():
                probe.before()
            output, timing = measure(synth(request), device, disable_gc=m.disable_gc)
            telemetry = {name: probe.after() for name, probe in probes.items()}
            waveform, to_host_ms = host_timed(lambda out=output: to_numpy(out.audio))

            audio_seconds = len(waveform) / output.sample_rate
            key = audio_key(prompt.id, job.nfe, repeat)
            if _should_keep_audio(m.save_audio, repeat):
                audio[key] = encode_wav(waveform, output.sample_rate)
            samples.append(
                Sample(
                    prompt_id=prompt.id,
                    nfe=job.nfe,
                    repeat=repeat,
                    seed=seed,
                    latency_ms=timing.latency_ms,
                    device_ms=timing.device_ms,
                    to_host_ms=to_host_ms,
                    stages_ms=timing.stages_ms,
                    stage_calls=timing.stage_calls,
                    audio_seconds=audio_seconds,
                    sample_rate=output.sample_rate,
                    rtf=(timing.latency_ms / 1000) / audio_seconds if audio_seconds else float("inf"),
                    audio_sha256=fingerprint(waveform),
                    audio_path=f"audio/{key}.wav" if key in audio else None,
                    telemetry=telemetry,
                )
            )

    return JobResult(
        nfe=job.nfe,
        samples=samples,
        audio=audio,
        worker=WorkerInfo(
            environment=worker_environment(),
            model_info=model.info(),
            model_load_ms=load_ms,
            cold_start=load_ms is not None,
            warmup_ms=tuple(warmup_ms),
        ),
    )


def execute_job_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Dict-in/dict-out entrypoint for remote backends (avoids pickling pydantic classes)."""
    return execute_job(Job.model_validate(payload)).model_dump()
