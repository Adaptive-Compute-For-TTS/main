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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

ASR_MODEL = "openai/whisper-small"  # what Arush's WER used; reconcile before the dashboard shows it

# Same idea as Arush's scorer: lowercase, drop punctuation. His also expands contractions and strips a
# re-spoken reference intro; neither is done here, so a WER from this module can differ slightly from
# wer_arush.csv until the two are merged.
_PUNCT = re.compile(r"[^\w\s$.%']|(?<!\d)\.|\.(?!\d)")


def normalize(text: str) -> List[str]:
    return _PUNCT.sub(" ", text.lower()).split()


def _edit_distance(ref: Sequence, hyp: Sequence) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def wer_cer(reference: str, hypothesis: str) -> Dict[str, float]:
    """Word and character error rate against the prompt text, each capped at 1 (as in wer_arush.csv)."""
    ref_w, hyp_w = normalize(reference), normalize(hypothesis)
    ref_c, hyp_c = " ".join(ref_w), " ".join(hyp_w)
    if not ref_w:
        raise ValueError("reference text has no words")
    return {
        "wer": min(1.0, _edit_distance(ref_w, hyp_w) / len(ref_w)),
        "cer": min(1.0, _edit_distance(ref_c, hyp_c) / len(ref_c)),
    }


def _whisper_transcriber(model_id: str = ASR_MODEL) -> Callable[[str], str]:
    from transformers import pipeline  # lazy: heavy, and not needed when a transcriber is injected

    asr = pipeline("automatic-speech-recognition", model=model_id)
    return lambda path: asr(str(path))["text"]


class QualityEvaluator:
    def __init__(
        self,
        reference_wav: Optional[str] = None,
        sim: Any = None,
        transcribe: Optional[Callable[[str], str]] = None,
    ) -> None:
        """reference_wav: the voice being cloned (None skips sim_ref).
        sim: object with .similarity(a, b) -> float (default: speaker_sim.SpeakerSim, loaded on first use).
        transcribe: path -> text (default: Whisper small, loaded on first use)."""
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
            self._transcribe = _whisper_transcriber()
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
            "asr_model": ASR_MODEL,
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
