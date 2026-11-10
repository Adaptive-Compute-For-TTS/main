"""Tests for score_demo.py. No Whisper: transcripts come from a pre-filled cache."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from asr import ASR_SETTINGS  # noqa: E402
from wer_scoring import score_clip  # noqa: E402
from score_demo import clip_key, corpus, score_entry, sha256  # noqa: E402

KETTLE = "The kettle is already on the stove."
INTRO = "Hey, Adam here. Let's create something that feels real, sounds human, and connects every time."


def item(text=KETTLE, run_id="r1", prompt_id="p001", nfe=20):
    return {"run_id": run_id, "prompt_id": prompt_id, "nfe": nfe, "text": text, "audio": "a.wav"}


def asr(transcript, status="ok"):
    return {"audio_sha256": "x", "transcript": transcript, "asr_status": status, "duration_s": 3.0}


def test_key_includes_nfe():
    assert clip_key("xvector_4090", "p001", 4) == "xvector_4090__p001__nfe4"
    assert clip_key("r", "p001", "20") != clip_key("r", "p001", "32")


def test_entry_matches_score_clip_no_intro():
    hyp = "the kettle is already on the stone"
    e = score_entry(item(), asr(hyp))
    s = score_clip(KETTLE, hyp)
    assert e["wer"] == pytest.approx(s["wer_no_intro"], abs=1e-4) == pytest.approx(1 / 7, abs=1e-4)
    assert e["cer"] == pytest.approx(s["cer_no_intro"], abs=1e-4)
    assert (e["word_errors"], e["ref_words"], e["subs"]) == (1, 7, 1)
    assert e["valid"] and e["flags"] == []


def test_rejected_clip_kept_as_one_and_invalid():
    e = score_entry(item(), asr("", "rejected: silent (RMS 0.00000 < 0.001)"))
    assert e["wer"] == e["cer"] == 1.0
    assert e["word_errors"] == e["ref_words"] == 7
    assert e["valid"] is False and e["flags"][0].startswith("rejected")


def test_intro_is_set_aside_and_flagged():
    e = score_entry(item(), asr(f"{INTRO} {KETTLE}"))
    assert e["wer"] == 0.0
    assert "intro_detected" in e["flags"]


def test_wer_is_uncapped():
    e = score_entry(item("Hello."), asr("hello there my good friend"))
    assert e["wer"] == 4.0


def test_corpus_is_total_over_total_not_mean():
    a = score_entry(item("Hello there.", prompt_id="p1"), asr("hello"))      # 1 error / 2 words
    b = score_entry(item(" ".join(["word"] * 48) + ".", prompt_id="p2"), asr(" ".join(["word"] * 48)))
    c = corpus([a, b])
    assert c["wer"] == pytest.approx(1 / 50, abs=1e-4)                      # mean would be 0.25
    assert (c["clips"], c["valid_clips"], c["word_errors"], c["ref_words"]) == (2, 2, 1, 50)


def run_cli(*args):
    return subprocess.run([sys.executable, str(ROOT / "score_demo.py"), *map(str, args)],
                          capture_output=True, text=True, cwd=ROOT)


@pytest.fixture
def wav_with_cache(tmp_path):
    """A synthetic WAV plus a cached transcript for it, so the CLI never loads Whisper."""
    wav = tmp_path / "clip.wav"
    sf.write(wav, 0.1 * np.sin(np.linspace(0, 2000, 48000)), 48000)
    out = tmp_path / "out" / "wer.json"
    out.parent.mkdir()
    cache = {"audio_sha256": sha256(wav), "transcript": "the kettle is already on the stone",
             "asr_status": "ok", "duration_s": 1.0, "asr": dict(ASR_SETTINGS)}
    (out.parent / f"transcripts_{ASR_SETTINGS['model']}.jsonl").write_text(json.dumps(cache) + "\n")
    return wav, out


def test_cli_single_wav_then_merge(wav_with_cache):
    wav, out = wav_with_cache
    r = run_cli("--wav", wav, "--text", KETTLE, "--run-id", "r1", "--prompt-id", "p001", "--nfe", 20, "--out", out)
    assert r.returncode == 0, r.stderr
    assert "cached" in r.stdout
    r = run_cli("--wav", wav, "--text", KETTLE, "--run-id", "r1", "--prompt-id", "p001", "--nfe", 32, "--out", out)
    assert r.returncode == 0, r.stderr
    d = json.loads(out.read_text())
    assert list(d["clips"]) == ["r1__p001__nfe20", "r1__p001__nfe32"]
    assert d["corpus"]["word_errors"] == 2 and d["corpus"]["ref_words"] == 14
    assert set(d["by_nfe"]) == {"20", "32"}
    assert d["asr"] == ASR_SETTINGS


def test_cli_rerun_replaces_same_key(wav_with_cache):
    wav, out = wav_with_cache
    for _ in range(2):
        r = run_cli("--wav", wav, "--text", KETTLE, "--run-id", "r1", "--prompt-id", "p001", "--nfe", 4, "--out", out)
        assert r.returncode == 0, r.stderr
    assert len(json.loads(out.read_text())["clips"]) == 1


def test_cli_manifest(wav_with_cache, tmp_path):
    wav, out = wav_with_cache
    m = tmp_path / "manifest.csv"
    m.write_text(f"audio_path,text,prompt_id,nfe\n{wav.name},{KETTLE},p001,4\n{wav.name},{KETTLE},p001,20\n")
    r = run_cli("--manifest", m, "--run-id", "xvector_4090", "--out", out)
    assert r.returncode == 0, r.stderr
    assert list(json.loads(out.read_text())["clips"]) == ["xvector_4090__p001__nfe20", "xvector_4090__p001__nfe4"]


def test_cli_refuses_other_asr_model(wav_with_cache):
    wav, out = wav_with_cache
    out.write_text(json.dumps({"asr": {**ASR_SETTINGS, "model": "large-v3"}, "clips": {}}))
    r = run_cli("--wav", wav, "--text", KETTLE, "--run-id", "r1", "--prompt-id", "p001", "--nfe", 4, "--out", out)
    assert r.returncode == 2 and "different ASR settings" in r.stderr


def test_cli_bad_input(wav_with_cache, tmp_path):
    wav, out = wav_with_cache
    r = run_cli("--wav", wav, "--text", KETTLE, "--prompt-id", "p001", "--nfe", 4, "--out", out)
    assert r.returncode == 2 and "--run-id" in r.stderr
    r = run_cli("--wav", tmp_path / "missing.wav", "--text", KETTLE, "--run-id", "r", "--prompt-id", "p", "--nfe", 4,
                "--out", out)
    assert r.returncode == 2 and "audio not found" in r.stderr
    m = tmp_path / "dup.csv"
    m.write_text(f"audio_path,text,prompt_id,nfe\n{wav.name},{KETTLE},p001,4\n{wav.name},{KETTLE},p001,4\n")
    r = run_cli("--manifest", m, "--run-id", "r", "--out", out)
    assert r.returncode == 2 and "duplicate key" in r.stderr
