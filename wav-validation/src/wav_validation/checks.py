"""The checks: measure a clip and decide whether it is usable.

Thresholds are the team's existing numbers: the too-short / silent rules from
quality-evals/speaker_sim (the SIM scorer refuses those clips) and the signal rules from the
benchmark's flag_wav_runs.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


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
