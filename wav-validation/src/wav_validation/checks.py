"""The checks: measure a clip and decide whether it is usable.

Thresholds are the team's existing numbers: the too-short / silent rules from
quality-evals/speaker_sim (the SIM scorer refuses those clips) and the signal rules from the
benchmark's flag_wav_runs.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import soundfile as sf

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

    Raises ValueError for input that is a caller bug rather than bad audio:
      - a torch tensor left as (1, frames): call .squeeze() first, as gpu-executor/scripts/generate.py does
      - integer samples (e.g. raw int16 PCM): the checks expect float samples in [-1, 1], and int16
        values would pass as "ok" with a peak in the thousands. Read files with
        soundfile.read(..., dtype="float32"), which scales them.
    """
    raw = np.asarray(audio)
    if raw.dtype.kind != "f":
        raise ValueError(f"expected float samples in [-1, 1], got dtype {raw.dtype}")
    samples = raw.astype(np.float32, copy=False)
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


def _r(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(float(x), 6)


def validate_audio(audio: Any, sample_rate: int, *, num_patches: Optional[int] = None,
                   max_generate_length: Optional[int] = None, thresholds: Thresholds = DEFAULT) -> Dict[str, Any]:
    """Check one generated clip in memory. This is what the GPU worker calls right after generation.

    audio: 1-D samples or (frames, channels), e.g. result.output["audio"].float().cpu().squeeze().numpy()
    num_patches / max_generate_length: from the executor and the generation config; enable the
    cut-off check (see cap_check).

    Returns plain JSON types (draft shape for the run's result["validation"]):
      ok          False if any flag is "invalid": fail the run, don't serve or score the clip
      flags       every flag that applies, in FLAGS order
      signal_ok   no caution flags (None when the clip couldn't be measured)
      checks      the measurements behind the flags
      thresholds  the numbers used
      version     wav-validation version
    """
    from wav_validation import __version__

    if isinstance(sample_rate, bool) or int(sample_rate) != sample_rate or sample_rate <= 0:
        raise ValueError(f"sample_rate must be a positive integer, got {sample_rate!r}")
    sample_rate = int(sample_rate)
    t = thresholds
    mono, channels = to_mono(audio)
    duration_s = mono.size / sample_rate
    non_finite = count_non_finite(mono)
    flags = []
    peak = rms = None
    measured: Dict[str, Any] = {"clipped_samples": None, "leading_silence_s": None,
                                "trailing_silence_s": None, "longest_gap_s": None}

    # Content checks short-circuit like build_run_json.audio_flags: there is no point measuring
    # silence layout in a clip that is broken, too short or silent.
    if non_finite:
        flags.append("non_finite")
    elif is_too_short(mono.size, sample_rate, t):
        flags.append("too_short")
    else:
        peak, rms = peak_and_rms(mono)
        if is_silent(peak, rms, t):
            flags.append("silent")
        else:
            measured = signal_measurements(mono, sample_rate, t)
            flags += caution_flags(peak, measured, t)

    cap = cap_check(duration_s, num_patches, max_generate_length, t)
    if cap["at_cap"]:
        flags.append("truncated_at_cap")

    order = list(FLAGS)
    flags.sort(key=order.index)
    return {
        "ok": not any(FLAGS[f][0] == INVALID for f in flags),
        "flags": flags,
        "signal_ok": None if measured["clipped_samples"] is None else is_signal_ok(flags),
        "checks": {
            # Full precision: rounding here and again in a report double-rounds (0.15155029 -> 0.15155
            # -> 0.1515 instead of 0.1516). The silence values below are whole frames, so rounding
            # them only removes float noise like 0.30000000000000004.
            "duration_s": float(duration_s),
            "sample_rate": sample_rate,
            "channels": channels,
            "peak": peak,
            "rms": rms,
            "non_finite_samples": non_finite,
            "clipped_samples": measured["clipped_samples"],
            "leading_silence_s": _r(measured["leading_silence_s"]),
            "trailing_silence_s": _r(measured["trailing_silence_s"]),
            "longest_gap_s": _r(measured["longest_gap_s"]),
            "num_patches": None if num_patches is None else int(num_patches),
            "max_generate_length": None if max_generate_length is None else int(max_generate_length),
            **cap,
        },
        "thresholds": t.as_dict(),
        "version": __version__,
    }


def _file_problem(flag: str, num_patches: Optional[int], max_generate_length: Optional[int],
                  t: Thresholds) -> Dict[str, Any]:
    """Result for a file that couldn't be read: same keys as validate_audio, nothing measured."""
    from wav_validation import __version__

    checks = {key: None for key in (
        "duration_s", "sample_rate", "channels", "peak", "rms", "non_finite_samples", "clipped_samples",
        "leading_silence_s", "trailing_silence_s", "longest_gap_s", "cap_s", "at_cap", "cap_source")}
    checks["num_patches"] = None if num_patches is None else int(num_patches)
    checks["max_generate_length"] = None if max_generate_length is None else int(max_generate_length)
    return {"ok": False, "flags": [flag], "signal_ok": None, "checks": checks,
            "thresholds": t.as_dict(), "version": __version__}


def validate_wav(path: Union[str, Path], *, num_patches: Optional[int] = None,
                 max_generate_length: Optional[int] = None, thresholds: Thresholds = DEFAULT) -> Dict[str, Any]:
    """validate_audio() for a file on disk (CLI, tests, offline checks). Never raises for bad files:
    a missing or undecodable file comes back as missing_audio / unreadable_audio."""
    path = Path(path)
    if not path.exists():
        return _file_problem("missing_audio", num_patches, max_generate_length, thresholds)
    try:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception:  # soundfile raises its own error types for corrupt or non-audio files
        return _file_problem("unreadable_audio", num_patches, max_generate_length, thresholds)
    return validate_audio(audio, sample_rate, num_patches=num_patches,
                          max_generate_length=max_generate_length, thresholds=thresholds)
