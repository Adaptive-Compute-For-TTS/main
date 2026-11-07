"""The checks: measure a clip and decide whether it is usable.

Thresholds are the team's existing numbers: the too-short / silent rules from
quality-evals/speaker_sim (the SIM scorer refuses those clips) and the signal rules from the
benchmark's flag_wav_runs.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

MAX_CHANNELS = 16


@dataclass(frozen=True)
class Thresholds:
    # Invalid: the clip can't be used (same rules as quality-evals/speaker_sim/speaker_sim.py).
    min_seconds: float = 0.5
    min_peak: float = 0.01  # about -40 dBFS
    min_rms: float = 1e-4

    # Caution: worth listening to, but the run still succeeds (flag_wav_runs.py signal_ok).
    max_peak: float = 0.95
    clip_level: float = 0.999
    max_leading_silence_s: float = 0.5
    min_trailing_silence_s: float = 0.05
    max_internal_gap_s: float = 1.0

    # How silence is measured: 20 ms frames, "silent" = more than 40 dB below the loudest frame.
    frame_s: float = 0.02
    silence_below_peak_db: float = 40.0

    # Generation length cap: one latent patch is 0.16 s of 48 kHz audio (patch 4 x hop 1920).
    patch_seconds: float = 0.16
    cap_tolerance_s: float = 0.01

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT = Thresholds()


INVALID = "invalid"  # the run fails; the clip is not stored as a result or scored
CAUTION = "caution"  # reported, but the run still succeeds

# flag -> (severity, description). Order is the order flags are listed on a clip. Names shared with
# quality-evals/speaker_sim/build_run_json.py mean the same thing there.
FLAGS: Dict[str, tuple] = {
    "missing_audio": (INVALID, "The WAV file is not on disk."),
    "unreadable_audio": (INVALID, "The WAV file could not be decoded."),
    "non_finite": (INVALID, "Contains NaN or infinite samples (a numerical blow-up during generation)."),
    "too_short": (INVALID, f"Shorter than {DEFAULT.min_seconds} s."),
    "silent": (INVALID, f"(Near-)silent: peak below {DEFAULT.min_peak} or RMS below {DEFAULT.min_rms}."),
    "truncated_at_cap": (INVALID, "Stopped at the generation length cap (max_generate_length), "
                                  "usually mid-sentence."),
    "clipped": (CAUTION, f"Has samples at full scale (|x| >= {DEFAULT.clip_level})."),
    "too_loud": (CAUTION, f"Peak above {DEFAULT.max_peak}."),
    "leading_silence": (CAUTION, f"Starts with {DEFAULT.max_leading_silence_s} s or more of silence."),
    "abrupt_end": (CAUTION, f"Less than {DEFAULT.min_trailing_silence_s} s of silence at the end: "
                            "speech may be cut off."),
    "long_gap": (CAUTION, f"A pause longer than {DEFAULT.max_internal_gap_s} s in the middle."),
}


def to_mono(audio: Any) -> Tuple[np.ndarray, int]:
    """(mono float32 samples, channel count) from 1-D audio or soundfile's (frames, channels) layout.

    Raises ValueError for shapes that are a caller bug rather than bad audio, e.g. a torch tensor
    left as (1, frames): call .squeeze() first, as gpu-executor/scripts/generate.py does.
    """
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 1:
        return samples, 1
    if samples.ndim != 2:
        raise ValueError(f"expected 1-D or (frames, channels) audio, got shape {samples.shape}")
    channels = samples.shape[1]
    if channels == 0 or channels > MAX_CHANNELS:
        raise ValueError(f"audio shape {samples.shape} looks like (channels, frames); "
                         "pass 1-D or (frames, channels)")
    if channels == 1:
        return samples[:, 0], 1
    return samples.mean(axis=1, dtype=np.float32), channels


def count_non_finite(mono: np.ndarray) -> int:
    """NaN/inf samples. Float32 WAVs (what the executor writes) can carry them; int PCM can't."""
    return int(mono.size - np.count_nonzero(np.isfinite(mono)))


def is_too_short(n_samples: int, sample_rate: int, t: Thresholds = DEFAULT) -> bool:
    """Same rule as speaker_sim.py: fewer samples than min_seconds of audio."""
    return n_samples < sample_rate * t.min_seconds


def peak_and_rms(mono: np.ndarray) -> Tuple[float, float]:
    """(max |x|, root-mean-square); (0, 0) for empty audio."""
    if mono.size == 0:
        return 0.0, 0.0
    return float(np.abs(mono).max()), float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))


def is_silent(peak: float, rms: float, t: Thresholds = DEFAULT) -> bool:
    """Same rule as speaker_sim.py / build_run_json.py: a failed generation, not a quiet voice."""
    return peak < t.min_peak or rms < t.min_rms


def signal_measurements(mono: np.ndarray, sample_rate: int, t: Thresholds = DEFAULT) -> Dict[str, Any]:
    """Clipping and silence layout, with the same math as flag_wav_runs.py:signal_checks.

    The clip is cut into frame_s frames; a frame is "voiced" when its level is within
    silence_below_peak_db of the loudest frame. Leading/trailing silence and the longest gap are
    counted in unvoiced frames. Audio shorter than one frame has no frames and reports zeros.
    """
    clipped = int((np.abs(mono) >= t.clip_level).sum())
    frame = int(t.frame_s * sample_rate)
    frame_count = len(mono) // frame if frame > 0 else 0
    if frame_count == 0:
        return {"clipped_samples": clipped, "leading_silence_s": 0.0, "trailing_silence_s": 0.0,
                "longest_gap_s": 0.0}

    frames = mono[: frame_count * frame].reshape(frame_count, frame)
    level_db = 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-9)
    voiced = np.flatnonzero(level_db > level_db.max() - t.silence_below_peak_db)

    longest_gap = 0
    if voiced.size > 1:
        longest_gap = int((np.diff(voiced) - 1).max())
    return {
        "clipped_samples": clipped,
        "leading_silence_s": float(voiced[0] * t.frame_s) if voiced.size else 0.0,
        "trailing_silence_s": float((frame_count - 1 - voiced[-1]) * t.frame_s) if voiced.size else 0.0,
        "longest_gap_s": float(longest_gap * t.frame_s),
    }


def caution_flags(peak: float, m: Dict[str, Any], t: Thresholds = DEFAULT) -> list:
    """Caution flags from peak + signal_measurements(); together they are flag_wav_runs.py's signal_ok."""
    flags = []
    if m["clipped_samples"] > 0:
        flags.append("clipped")
    if peak > t.max_peak:
        flags.append("too_loud")
    if m["leading_silence_s"] >= t.max_leading_silence_s:
        flags.append("leading_silence")
    if m["trailing_silence_s"] < t.min_trailing_silence_s:
        flags.append("abrupt_end")
    if m["longest_gap_s"] > t.max_internal_gap_s:
        flags.append("long_gap")
    return flags


def cap_check(duration_s: float, num_patches: Optional[int], max_generate_length: Optional[int],
              t: Thresholds = DEFAULT) -> Dict[str, Any]:
    """Did generation stop because it hit max_generate_length (so the speech is likely cut off)?

    With num_patches from the executor (generate_fixed_nfe(...).num_patches) the answer is exact.
    Without it, fall back to flag_wav_runs.py's rule: duration within cap_tolerance_s of the cap.
    Without max_generate_length there is nothing to compare against, so at_cap is None.
    """
    if max_generate_length is None:
        return {"cap_s": None, "at_cap": None, "cap_source": None}
    cap_s = max_generate_length * t.patch_seconds
    if num_patches is not None:
        at_cap, source = num_patches >= max_generate_length, "patches"
    else:
        at_cap, source = duration_s >= cap_s - t.cap_tolerance_s, "duration"
    return {"cap_s": round(cap_s, 6), "at_cap": bool(at_cap), "cap_source": source}


def is_signal_ok(flags: list) -> bool:
    """The old wav_flags.csv `signal_ok` column: no caution flag."""
    return not any(FLAGS[f][0] == CAUTION for f in flags)
