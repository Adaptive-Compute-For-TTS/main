from __future__ import annotations

from pathlib import Path

import pytest

from ttsbench.audio import decode_wav
from ttsbench.schema import ComponentSpec, Job, MeasurementConfig, Prompt
from ttsbench.worker import execute_job, execute_job_payload


def _job(prompts: list[Prompt], **measurement: object) -> Job:
    return Job(
        model=ComponentSpec(name="sine"),
        nfe=4,
        prompts=tuple(prompts),
        measurement=MeasurementConfig(**{"warmup": 1, "repeats": 3, **measurement}),
    )


def test_execute_job_produces_one_sample_per_prompt_and_repeat(prompts: list[Prompt]) -> None:
    result = execute_job(_job(prompts))

    assert len(result.samples) == len(prompts) * 3
    assert len(result.worker.warmup_ms) == 1
    first = result.samples[0]
    assert first.stage_calls["flow_step"] == 4
    assert first.rtf == pytest.approx(first.latency_ms / 1000 / first.audio_seconds)
    assert first.latency_ms > 0 and first.to_host_ms >= 0


def test_fixed_seed_gives_bit_identical_audio(prompts: list[Prompt]) -> None:
    samples = execute_job(_job(prompts[:1])).samples
    assert len({s.audio_sha256 for s in samples}) == 1


def test_vary_seed_changes_audio(prompts: list[Prompt]) -> None:
    samples = execute_job(_job(prompts[:1], vary_seed=True)).samples
    assert [s.seed for s in samples] == [0, 1, 2]
    assert len({s.audio_sha256 for s in samples}) == 3


@pytest.mark.parametrize(("policy", "expected"), [("all", 3), ("first", 1), ("none", 0)])
def test_save_audio_policy(prompts: list[Prompt], policy: str, expected: int) -> None:
    result = execute_job(_job(prompts[:1], save_audio=policy))
    assert len(result.audio) == expected
    assert sum(s.audio_path is not None for s in result.samples) == expected


def test_saved_audio_roundtrips(prompts: list[Prompt]) -> None:
    result = execute_job(_job(prompts[:1]))
    audio, sr = decode_wav(next(iter(result.audio.values())))
    assert sr == result.samples[0].sample_rate
    assert len(audio) / sr == pytest.approx(result.samples[0].audio_seconds)


def test_model_is_loaded_once_per_process(prompts: list[Prompt]) -> None:
    job = _job(prompts[:1]).model_copy(update={"model": ComponentSpec(name="sine", params={"work": 32})})
    assert execute_job(job).worker.cold_start is True
    warm = execute_job(job).worker
    assert warm.cold_start is False and warm.model_load_ms is None


def test_missing_reference_audio_fails_loudly(prompts: list[Prompt], tmp_path: Path) -> None:
    cloned = prompts[0].model_copy(update={"reference_audio": "references/missing.wav"})
    with pytest.raises(FileNotFoundError, match="missing.wav"):
        execute_job(_job([cloned]), assets_root=tmp_path)


def test_payload_entrypoint_is_dict_in_dict_out(prompts: list[Prompt]) -> None:
    out = execute_job_payload(_job(prompts[:1]).model_dump())
    assert isinstance(out, dict) and out["nfe"] == 4 and len(out["samples"]) == 3
