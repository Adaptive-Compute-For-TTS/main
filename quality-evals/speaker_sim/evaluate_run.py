"""Quality Evaluation for one run: the box between WAV Validation and the database.

The GPU worker hands over a clip that already passed WAV validation. This module scores it and
returns one JSON-safe dict for the run's `result["quality"]`:

    from evaluate_run import QualityEvaluator
    qe = QualityEvaluator(reference_wav="abhi1.wav")          # loads models on first use, keep one
    quality = qe.evaluate(audio_path, prompt, validation=v)   # never raises

The SIM and ASR models are injectable so tests (and CI) need neither torch nor weights.

Not the same as the sweep tooling (build_run_json.py): one run has no "same prompt at the highest
NFE" clip, so there is no sim_hi_nfe here, and the run-wide latency_outlier caution does not apply.
"""
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# One scorer for the whole project: Arush's quality-evals/wer_cer, the one behind wer_arush.csv, so
# per-run and sweep numbers match. Needs quality-evals/wer_cer/requirements.txt installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wer_cer"))
import asr  # noqa: E402  (faster-whisper; the model itself loads on the first transcribe)
from wer_scoring import official_columns, score_clip  # noqa: E402


def wer_cer(reference: str, hypothesis: str) -> Dict[str, float]:
    """Word and character error rate against the prompt text (no-intro version, capped at 1, as in wer_arush.csv)."""
    # Checked on the raw text: the normalizer turns "..." into ".", so checking after it would not catch it.
    if not re.search(r"\w", reference):
        raise ValueError("reference text has no words")
    s = official_columns(score_clip(reference, hypothesis), "no_intro")
    return {"wer": s["wer"], "cer": s["cer"]}


def asr_model_name() -> str:
    """Read at call time, since asr.set_model() can switch small -> large-v3."""
    return f"faster-whisper-{asr.ASR_SETTINGS['model']}"


class QualityEvaluator:
    def __init__(
        self,
        reference_wav: Optional[str] = None,
        sim: Any = None,
        transcribe: Optional[Callable[[str], str]] = None,
    ) -> None:
        """reference_wav: the voice being cloned (None skips sim_ref).
        sim: object with .similarity(a, b) -> float (default: speaker_sim.SpeakerSim, loaded on first use).
        transcribe: path -> text (default: asr.transcribe, Whisper loaded on first use; it raises
        asr.AudioRejected for clips too short or silent, recorded in `errors` like any ASR failure)."""
        self.reference_wav = reference_wav
        self._sim = sim
        self._transcribe = transcribe

    def _get_sim(self) -> Any:
        if self._sim is None:
            from speaker_sim import SpeakerSim

            self._sim = SpeakerSim()
        return self._sim

    def _get_transcribe(self) -> Callable[[str], str]:
        if self._transcribe is None:
            self._transcribe = asr.transcribe
        return self._transcribe

    def evaluate(
        self,
        audio_path: str,
        prompt: str,
        validation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Score one clip. Always returns a dict: a failed metric is None plus an entry in `errors`,
        so one bad scorer cannot fail the whole run. A clip that failed validation is not scored."""
        try:
            from speaker_sim import MODEL_ID, MODEL_REVISION  # constants only; no model is loaded
            sim_model: Optional[str] = f"{MODEL_ID}@{MODEL_REVISION}"
        except ImportError:  # torch not installed (tests, CI) and a fake scorer was injected
            sim_model = None

        out: Dict[str, Any] = {
            "scored": False,
            "sim_ref": None,
            "wer": None,
            "cer": None,
            "transcript": None,
            "sim_model": sim_model,
            "asr_model": asr_model_name(),
            "errors": [],
        }
        if validation is not None and not validation.get("ok", False):
            out["errors"].append("skipped: clip failed WAV validation: " + ", ".join(validation.get("flags", [])))
            return out
        if not Path(audio_path).is_file():
            out["errors"].append(f"audio not found: {Path(audio_path).name}")
            return out

        if self.reference_wav:
            try:
                out["sim_ref"] = round(float(self._get_sim().similarity(audio_path, self.reference_wav)), 4)
            except Exception as error:
                out["errors"].append(f"sim_ref: {type(error).__name__}: {error}")
        try:
            transcript = self._get_transcribe()(audio_path)
            out["transcript"] = transcript
            out.update({k: round(v, 4) for k, v in wer_cer(prompt, transcript).items()})
        except Exception as error:
            out["errors"].append(f"asr: {type(error).__name__}: {error}")

        out["scored"] = out["sim_ref"] is not None or out["wer"] is not None
        return out
