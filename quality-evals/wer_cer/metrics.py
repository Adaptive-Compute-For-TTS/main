"""WER (with format-only normalization rules) for the dots.tts eval pipeline (SIM lives in ../speaker_sim).

Debug a normalization mismatch:
    python metrics.py "Your total comes to $42.75." "your total comes to forty two dollars and seventy five cents"
"""
import re
import sys

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

_whisper_norm = EnglishTextNormalizer()

# ---------------------------------------------------------------------------
# Custom normalization rules
#
# GOLDEN RULE: a rule may only reconcile different ways of WRITING the same
# spoken words ("Ave." vs "Avenue", "a.m." vs "AM"). Never add a rule that turns
# a wrong word into a right one (e.g. correcting Whisper's spelling of "Przybylski").
# That would hide real TTS mistakes too.
#
# Each rule is (regex, replacement), applied to BOTH reference and transcript.
# Every new rule should come with a written/spoken pair in the test suite.
# ---------------------------------------------------------------------------

# Applied to lowercased raw text, BEFORE Whisper's normalizer.
PRE_RULES = [
    # Dotted acronyms and times: "a.m." -> "am", "t.t.s." -> "tts"      (p011, p016, p036)
    (r"\b((?:[a-z]\.){2,})", lambda m: m.group(1).replace(".", "")),
    # Clock times: "6:15" and "6.15 am" -> "615"                         (p011, p016)
    (r"(\d):(\d)", r"\1\2"),
    (r"(\d)\.(\d\d)\s*(am|pm)\b", r"\1\2 \3"),
    # Digit groups whose second group starts with 0: "555-0147" / "555 0147" -> "5550147".
    # Must run before Whisper's normalizer, which drops the leading zero of a separate
    # number ("0147" -> "147") but keeps it inside one ("5550147")      (p012)
    (r"(\d)[ -](?=0\d)", r"\1"),
    # Mixed fractions: "2 1/2" / "2½" -> "2.5"                           (p014)
    (r"\b(\d+)\s+1/2\b", r"\1.5"),
    (r"\b(\d+)\s*½", r"\1.5"),
    # Street, room and month abbreviations                              (p033, p036)
    (r"\bave\b\.?", "avenue"),
    (r"\bhwy\b\.?", "highway"),
    (r"\brm\b\.?", "room"),
    # Months with or without the dot: "Jan." / "Jan" (Whisper drops it)  (p033; no prompt uses
    # "jan" or "mar" as another word)
    (r"\bjan\b\.?", "january"),
    (r"\bmar\b\.?", "march"),
    # Compounds Whisper writes differently                               (p044, p026)
    (r"\bdoor[ -]frame", "doorframe"),
    (r"\bquetzal[ -]?feather", "quetzal feather"),
    # "startup's" / "start-up's" / "start-ups" / "startups" -> "startups", before Whisper's
    # normalizer would turn "'s" into "is"                               (p035)
    (r"\bstart-?up(?:'s|s)\b", "startups"),
]

# Applied AFTER Whisper's normalizer.
POST_RULES = [
    # Digit groups split by spaces: "555 1234", "6 15" -> one token     (p012, p011)
    # but never into a following year: "March 3, 2024" stays "3 2024"   (p013)
    (r"(?<=\d) (?=\d)(?!(?:1[5-9]|20)\d\d\b)", ""),
    # "2 and a half" -> "2.5"                                            (p014)
    (r"\b(\d+) and a half\b", r"\1.5"),
]


def normalize(text):
    s = text.lower()
    for pattern, repl in PRE_RULES:
        s = re.sub(pattern, repl, s)
    s = _whisper_norm(s)
    for pattern, repl in POST_RULES:
        s = re.sub(pattern, repl, s)
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------

def compute_wer(reference, transcript):
    """Per-clip WER with substitution/deletion/insertion counts."""
    ref, hyp = normalize(reference), normalize(transcript)
    n_ref = len(ref.split())
    if not hyp:  # nothing transcribed: every reference word was deleted
        return {"wer": 1.0, "subs": 0, "dels": n_ref, "ins": 0, "ref_words": n_ref,
                "ref_norm": ref, "hyp_norm": hyp}
    out = jiwer.process_words(ref, hyp)
    return {
        "wer": out.wer,
        "subs": out.substitutions,
        "dels": out.deletions,
        "ins": out.insertions,
        "ref_words": out.hits + out.substitutions + out.deletions,
        "ref_norm": ref,
        "hyp_norm": hyp,
    }


def wer_cer(reference, transcript):
    """(WER, CER) for one clip, normalized the same way as wer_scoring.py's "full" scores.

    Uncapped, so insertions can push either above 1.0. An empty transcript scores (1.0, 1.0);
    an empty reference has no defined error rate and scores (nan, nan).
    """
    ref, hyp = normalize(reference), normalize(transcript)
    if not ref:
        return float("nan"), float("nan")
    if not hyp:
        return 1.0, 1.0
    return jiwer.wer(ref, hyp), jiwer.cer(ref, hyp)


def corpus_wer(rows):
    """Corpus WER = total errors / total reference words over many clips.

    `rows` are compute_wer() results (or per-clip CSV rows with the same keys).
    This equals jiwer on lists of references/transcripts, but also handles empty
    transcripts and works directly on grouped CSV rows (per bucket, per category).
    Never average per-clip WER instead.
    """
    rows = list(rows)
    errors = sum(r["subs"] + r["dels"] + r["ins"] for r in rows)
    words = sum(r["ref_words"] for r in rows)
    return {"wer": errors / words if words else float("nan"),
            "errors": errors, "ref_words": words, "clips": len(rows)}


if __name__ == "__main__":
    written, spoken = sys.argv[1], sys.argv[2]
    result = compute_wer(written, spoken)
    print("reference ->", result["ref_norm"])
    print("transcript->", result["hyp_norm"])
    print({k: v for k, v in result.items() if not k.endswith("_norm")})
