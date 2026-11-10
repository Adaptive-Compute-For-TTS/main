from __future__ import annotations

import pytest

from ttsbench.schema import Prompt, Sample
from ttsbench.summary import NOISY_CV, describe, summarize


def _sample(prompt_id: str, latency: float, sha: str = "h", repeat: int = 0) -> Sample:
    return Sample(
        prompt_id=prompt_id,
        nfe=4,
        repeat=repeat,
        seed=0,
        latency_ms=latency,
        to_host_ms=0.1,
        audio_seconds=1.0,
        sample_rate=24000,
        rtf=latency / 1000,
        audio_sha256=sha,
    )


def test_describe() -> None:
    stats = describe([1.0, 2.0, 3.0, 4.0])
    assert stats["n"] == 4 and stats["p50"] == 2.5 and stats["min"] == 1.0 and stats["max"] == 4.0
    assert stats["cv"] == pytest.approx(stats["std"] / 2.5)
    assert describe([]) == {"n": 0}


def test_noisy_uses_within_prompt_variance(prompts: list[Prompt]) -> None:
    # Very different prompts, each perfectly stable: not noisy.
    stable = [_sample("s1", 10.0, repeat=r) for r in range(3)] + [_sample("l1", 500.0, repeat=r) for r in range(3)]
    row = next(r for r in summarize(stable, prompts) if r["length"] == "all")
    assert row["within_prompt_cv"] == 0.0 and not row["noisy"]

    jittery = [_sample("s1", v, repeat=r) for r, v in enumerate((10.0, 14.0, 6.0))]
    row = next(r for r in summarize(jittery, prompts) if r["length"] == "all")
    assert row["within_prompt_cv"] > NOISY_CV and row["noisy"]


def test_groups_by_length_and_determinism(prompts: list[Prompt]) -> None:
    samples = [_sample("s1", 10.0, "a", 0), _sample("s1", 10.0, "b", 1), _sample("m1", 20.0, "c", 0)]
    rows = {r["length"]: r for r in summarize(samples, prompts)}
    assert set(rows) == {"all", "short", "medium"}
    assert rows["short"]["deterministic_fraction"] == 0.0
    assert rows["medium"]["deterministic_fraction"] == 1.0
    assert rows["all"]["n"] == 3
