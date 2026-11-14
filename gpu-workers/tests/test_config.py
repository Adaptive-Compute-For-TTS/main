from pathlib import Path

import pytest

from gpu_worker import WorkerConfig


def test_defaults_match_the_baseline():
    c = WorkerConfig.from_env({})
    assert c == WorkerConfig()
    assert (c.model, c.precision, c.optimize, c.seed) == ("dots-studio/dots.tts-soar", "bfloat16", False, 42)
    assert (c.guidance_scale, c.speaker_scale, c.warmup_nfes) == (1.2, 1.5, (10,))


def test_env_overrides():
    c = WorkerConfig.from_env({
        "GPU_WORKER_MODEL": "dots-studio/dots.tts-base", "GPU_WORKER_OPTIMIZE": "true",
        "GPU_WORKER_SEED": "7", "GPU_WORKER_GUIDANCE_SCALE": "1.5", "GPU_WORKER_STORAGE_DIR": "/data/wavs",
        "GPU_WORKER_WARMUP_NFES": "4, 10 32", "UNRELATED": "x",
    })
    assert c.model == "dots-studio/dots.tts-base" and c.optimize is True and c.seed == 7
    assert c.guidance_scale == 1.5 and c.storage_dir == Path("/data/wavs") and c.warmup_nfes == (4, 10, 32)


def test_empty_warmup_list_means_no_warmup():
    assert WorkerConfig.from_env({"GPU_WORKER_WARMUP_NFES": ""}).warmup_nfes == ()


def test_bad_boolean_is_an_error():
    with pytest.raises(ValueError):
        WorkerConfig.from_env({"GPU_WORKER_OPTIMIZE": "maybe"})


def test_reference_wav_is_optional():
    assert WorkerConfig.from_env({}).reference_wav is None
    assert WorkerConfig.from_env({"GPU_WORKER_REFERENCE_WAV": ""}).reference_wav is None
    assert WorkerConfig.from_env({"GPU_WORKER_REFERENCE_WAV": "/data/abhi1.wav"}).reference_wav == Path("/data/abhi1.wav")
