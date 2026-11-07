"""Checks each generated WAV before it is stored or scored.

    from wav_validation import validate_audio
    result = validate_audio(audio, sample_rate, num_patches=..., max_generate_length=...)
    if not result["ok"]: ...  # fail the run; result["flags"] says why
"""

__version__ = "0.1.0"

from wav_validation.checks import (  # noqa: E402  (after __version__, which checks.py reads)
    CAUTION,
    DEFAULT,
    FLAGS,
    INVALID,
    Thresholds,
    validate_audio,
    validate_wav,
)

__all__ = ["CAUTION", "DEFAULT", "FLAGS", "INVALID", "Thresholds", "validate_audio", "validate_wav",
           "__version__"]
