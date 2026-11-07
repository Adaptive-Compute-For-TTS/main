"""Checks each generated WAV before it is stored or scored.

    from wav_validation import validate_and_store
    result = validate_and_store(audio, sample_rate, run_id, storage_dir,
                                num_patches=..., max_generate_length=...)
    if not result["ok"]: ...  # fail the run; result["flags"] says why
    else: ...                 # save result["audio_path"], hand the clip to quality eval
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
from wav_validation.store import REJECTED_DIR, validate_and_store  # noqa: E402

__all__ = ["CAUTION", "DEFAULT", "FLAGS", "INVALID", "REJECTED_DIR", "Thresholds", "validate_and_store",
           "validate_audio", "validate_wav", "__version__"]
