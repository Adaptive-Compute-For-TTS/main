"""ASR for the dots.tts eval pipeline: load audio, reject unusable clips, transcribe.

Usage:  python asr.py path/to/p001.wav
"""
import sys
from functools import lru_cache

import numpy as np
from faster_whisper import WhisperModel, decode_audio

SAMPLE_RATE = 16_000   # Whisper and WavLM both expect 16 kHz mono
MIN_DURATION_S = 0.5   # clips shorter than this are rejected
MIN_RMS = 1e-3         # about -60 dBFS; anything quieter counts as silent

# Recorded in every results summary so runs are reproducible.
# Switch "model" to "large-v3" (and device/compute_type to "cuda"/"float16") for official GPU runs.
ASR_SETTINGS = {
    "model": "small",
    "device": "cpu",
    "compute_type": "int8",
    "language": "en",
    "beam_size": 5,
    "temperature": 0.0,
    "condition_on_previous_text": False,
}


class AudioRejected(ValueError):
    """Raised for clips that are too short or too quiet to score."""


def load_audio(path):
    """Read WAV/M4A/MP3/etc. and return mono float32 samples at 16 kHz."""
    return decode_audio(str(path), sampling_rate=SAMPLE_RATE)


def check_audio(audio, path=""):
    """Reject short or silent clips instead of scoring them. Returns duration in seconds."""
    duration = len(audio) / SAMPLE_RATE
    if duration < MIN_DURATION_S:
        raise AudioRejected(f"{path}: too short ({duration:.2f}s < {MIN_DURATION_S}s)")
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms < MIN_RMS:
        raise AudioRejected(f"{path}: silent (RMS {rms:.5f} < {MIN_RMS})")
    return duration


def set_model(name):
    """Switch the Whisper model (e.g. "large-v3"); every other setting stays the same.
    Updates ASR_SETTINGS in place, so every module that imported it records the new model."""
    ASR_SETTINGS["model"] = name
    get_model.cache_clear()


@lru_cache(maxsize=1)
def get_model():
    s = ASR_SETTINGS
    return WhisperModel(s["model"], device=s["device"], compute_type=s["compute_type"])


def transcribe(path):
    """Return Whisper's transcript of one clip. Raises AudioRejected for unusable clips."""
    audio = load_audio(path)
    check_audio(audio, path)
    s = ASR_SETTINGS
    segments, _ = get_model().transcribe(
        audio,
        language=s["language"],
        beam_size=s["beam_size"],
        temperature=s["temperature"],
        condition_on_previous_text=s["condition_on_previous_text"],
    )
    # segments is a generator: joining it is what actually runs the transcription
    return " ".join(seg.text.strip() for seg in segments).strip()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(f"{p}: {transcribe(p)}")
