"""Score Atharv's WER request (NFE benchmark clips Nick generated): transcribe every clip, then compute WER/CER two ways.

Transcripts are cached in asr_progress.jsonl next to clips.csv, and every score is recomputed
from those transcripts, so rescoring is cheap and never re-transcribes.

    python wer_scoring.py runs/nfe_benchmark/clips.csv --official no_intro --limit 10
    python wer_scoring.py runs/nfe_benchmark/clips.csv --official no_intro --files orig_3090__p036__nfe6.wav
    python wer_scoring.py runs/nfe_benchmark/clips.csv --official no_intro      # full run
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import jiwer
import pandas as pd

from asr import ASR_SETTINGS, SAMPLE_RATE, AudioRejected, check_audio, get_model, load_audio
from metrics import POST_RULES, PRE_RULES, corpus_wer, normalize

TEXT_COL = "text"   # the sentence each clip should say
PATH_COL = "file"   # e.g. orig_3090__p001__nfe4.wav, found in clips/ next to clips.csv
NICK_COLS = ["file", "run_id", "prompt_id", "nfe", "length", "content", "text", "repo_path", "wer"]

# Speech found at the start of the orig_3090 clips (Phase 2 test run). Stored raw and normalized
# with the same normalize() as references and transcripts.
INTRO_PHRASE_RAW = "Hey, Adam here. Let's create something that feels real, sounds human, and connects every time."
INTRO_WORDS = normalize(INTRO_PHRASE_RAW).split()

VERSIONS = ["full", "no_intro"]
METRICS = ["wer", "cer", "subs", "dels", "ins", "trailing_dels", "missing_end_words"]
HEARD = ["heard_ref_words", "heard_errors", "wer_heard"]
NEW_COLS = (["transcript", "cer", "wer_uncapped", "cer_uncapped", "wer_version", "missing_end_words"]
            + HEARD + ["intro_text"]
            + [f"{m}_{v}" for v in VERSIONS for m in METRICS]
            + ["ref_words", "ref_norm", "hyp_norm", "duration_s", "asr_status"])


def resolve_path(raw, base):
    """Try the path as given, relative to clips.csv, then clips/<filename>."""
    p = Path(str(raw))
    for cand in (p, base / p, base / "clips" / p.name):
        if cand.is_file():
            return cand
    return None


def transcribe_audio(path):
    """Expensive part: returns transcript, status and duration for one clip."""
    audio = load_audio(path)
    out = {"transcript": "", "asr_status": "ok", "duration_s": round(len(audio) / SAMPLE_RATE, 3)}
    try:
        check_audio(audio, path)
    except AudioRejected as e:
        out["asr_status"] = "rejected: " + str(e).split(": ", 1)[-1]
        return out
    s = ASR_SETTINGS
    segments, _ = get_model().transcribe(
        audio, language=s["language"], beam_size=s["beam_size"],
        temperature=s["temperature"], condition_on_previous_text=s["condition_on_previous_text"])
    out["transcript"] = " ".join(seg.text.strip() for seg in segments).strip()
    return out


# ---------------------------------------------------------------------------
# Scoring (cheap; runs on normalized text only)
# ---------------------------------------------------------------------------

def word_errors(ref_words, hyp_words):
    """S, D, I, trailing_dels and missing_end_words for word lists.

    trailing_dels:     reference words in a deletion chunk at the very end of the alignment.
    missing_end_words: reference words after the last reference word with a correct match (hit);
                       all of them if there is no hit. Unlike trailing_dels, this also catches
                       endings that jiwer aligns as substitutions (e.g. "stove" -> "st").
    """
    n = len(ref_words)
    if not hyp_words:  # nothing left to compare: every reference word was deleted
        return {"subs": 0, "dels": n, "ins": 0, "trailing_dels": n, "missing_end_words": n,
                "_last_hit": (0, 0)}
    out = jiwer.process_words(" ".join(ref_words), " ".join(hyp_words))
    chunks = out.alignments[0]
    last = chunks[-1] if chunks else None
    trailing = (last.ref_end_idx - last.ref_start_idx) if last and last.type == "delete" else 0
    hits = [c for c in chunks if c.type == "equal"]
    last_hit = (hits[-1].ref_end_idx, hits[-1].hyp_end_idx) if hits else (0, 0)
    return {"subs": out.substitutions, "dels": out.deletions, "ins": out.insertions,
            "trailing_dels": trailing, "missing_end_words": n - last_hit[0], "_last_hit": last_hit}


def heard_scores(ref_words, hyp_words):
    """WER over the part of the reference that was heard, i.e. ignoring a cut-off ending.

    heard_ref_words: reference words up to and including the last correctly heard word
                     (= ref_words - missing_end_words).
    heard_errors:    S + D + I between that part of the reference and the transcript words up to
                     the same hit; anything after the last hit is the cut-off tail and is ignored.
    wer_heard:       heard_errors / heard_ref_words; empty (NaN) when no word was heard.
    """
    r_end, h_end = word_errors(ref_words, hyp_words)["_last_hit"]
    if r_end == 0:
        return {"heard_ref_words": 0, "heard_errors": 0, "wer_heard": float("nan")}
    e = word_errors(ref_words[:r_end], hyp_words[:h_end])
    errors = e["subs"] + e["dels"] + e["ins"]
    return {"heard_ref_words": r_end, "heard_errors": errors, "wer_heard": errors / r_end}


def score_version(ref_words, hyp_words):
    """Uncapped WER/CER plus error counts for one (ref, hyp) pair."""
    e = word_errors(ref_words, hyp_words)
    ref, hyp = " ".join(ref_words), " ".join(hyp_words)
    if hyp:
        c = jiwer.process_characters(ref, hyp)
        char_errors, ref_chars = c.substitutions + c.deletions + c.insertions, c.hits + c.substitutions + c.deletions
    else:  # every reference character deleted
        char_errors = ref_chars = len(ref)
    e.pop("_last_hit")
    # char_errors / ref_chars are kept for corpus CER (not written to the CSV)
    # An empty reference has no defined error rate: NaN instead of dividing by zero
    wer = (e["subs"] + e["dels"] + e["ins"]) / len(ref_words) if ref_words else float("nan")
    cer = char_errors / ref_chars if ref_chars else float("nan")
    return {"wer": wer, "cer": cer, **e,
            **heard_scores(ref_words, hyp_words), "char_errors": char_errors, "ref_chars": ref_chars}


def split_intro(ref_words, hyp_words):
    """Find the transcript words that belong to the intro.

    Aligns the transcript against INTRO_WORDS + reference. The intro is the transcript up to and
    including the last word that exactly matches an intro word; words before that (substitutions
    or insertions between intro matches) belong to it too. Substitutions after the last exact
    intro match do not, so a misheard first word of the sentence ("Savan" for "Siobhan") is not
    taken as intro. Deleted intro positions are ignored, so a partial or missing intro is fine.
    Because alignments are monotonic, intro words are always a prefix.

    Returns (k, unattributed_lead): the transcript's first k words are the intro, and
    unattributed_lead counts the words after them that come before the first reference hit
    (all remaining words if there is no reference hit). A nonzero value can mean a different intro.
    """
    n_intro = len(INTRO_WORDS)
    chunks = jiwer.process_words(" ".join(INTRO_WORDS + ref_words), " ".join(hyp_words)).alignments[0]
    k, first_ref_hit = 0, len(hyp_words)
    for c in chunks:
        if c.type == "equal":
            for j in range(c.ref_end_idx - c.ref_start_idx):
                if c.ref_start_idx + j < n_intro:
                    k = c.hyp_start_idx + j + 1
                else:
                    first_ref_hit = min(first_ref_hit, c.hyp_start_idx + j)
    return k, max(first_ref_hit - k, 0)


def score_clip(reference, transcript):
    """Score one clip two ways from the same normalized reference and transcript.

    full:     the transcript as is (intro words count as errors).
    no_intro: transcript words attributed to INTRO_PHRASE by split_intro() are set aside in
              intro_text; the remaining words are scored against the reference only.
    An empty transcript (silent/rejected clip) scores WER = CER = 1.0 in both versions.
    """
    ref_norm, hyp_norm = normalize(reference), normalize(transcript)
    rw, hw = ref_norm.split(), hyp_norm.split()
    out = {"ref_words": len(rw), "ref_norm": ref_norm, "hyp_norm": hyp_norm, "intro_text": "",
           "unattributed_lead": 0}
    if not hw:
        silent = score_version(rw, [])
        for v in VERSIONS:
            out.update({f"{m}_{v}": silent[m] for m in METRICS + HEARD + ["char_errors", "ref_chars"]})
        return out

    k, out["unattributed_lead"] = split_intro(rw, hw)
    out["intro_text"] = " ".join(hw[:k])
    for v, words in (("full", hw), ("no_intro", hw[k:])):
        out.update({f"{m}_{v}": val for m, val in score_version(rw, words).items()})
    return out


def official_columns(scores, official):
    """Nick's wer/cer: the official version, capped at 1.0 and rounded; uncapped values kept."""
    w, c = scores[f"wer_{official}"], scores[f"cer_{official}"]
    return {"wer": round(min(w, 1.0), 4), "cer": round(min(c, 1.0), 4),
            "wer_uncapped": round(w, 4), "cer_uncapped": round(c, 4), "wer_version": official,
            "missing_end_words": scores[f"missing_end_words_{official}"],
            "heard_ref_words": scores[f"heard_ref_words_{official}"],
            "heard_errors": scores[f"heard_errors_{official}"],
            "wer_heard": round(scores[f"wer_heard_{official}"], 4)}


def version_rows(rows, v):
    """Rows in the shape corpus_wer() expects, for one scoring version."""
    return [{"subs": r[f"subs_{v}"], "dels": r[f"dels_{v}"], "ins": r[f"ins_{v}"],
             "ref_words": r["ref_words"]} for r in rows]


def corpus_cer(rows, v):
    """Corpus CER = total character errors / total reference characters (not the mean of per-clip CER)."""
    errors = sum(r[f"char_errors_{v}"] for r in rows)
    chars = sum(r[f"ref_chars_{v}"] for r in rows)
    return {"cer": errors / chars, "char_errors": errors, "ref_chars": chars, "clips": len(rows)}


def corpus_cer_table(df, scored):
    """Corpus CER for both versions: overall, by run_id, by nfe and by run_id x nfe."""
    rows = [{**s, "run_id": df["run_id"].iloc[i], "nfe": int(df["nfe"].iloc[i])} for i, s in enumerate(scored)]
    groups = {"all": {"all": rows}}
    for key in ("run_id", "nfe"):
        groups[key] = {}
        for r in rows:
            groups[key].setdefault(str(r[key]), []).append(r)
    groups["run_id x nfe"] = {}
    for r in rows:
        groups["run_id x nfe"].setdefault(f"{r['run_id']} nfe{r['nfe']}", []).append(r)
    return {name: {k: {v: corpus_cer(g, v) for v in VERSIONS} for k, g in grp.items()} for name, grp in groups.items()}


def pkg(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="path to clips.csv")
    ap.add_argument("--official", required=True, choices=VERSIONS,
                    help="which scoring version fills Nick's wer/cer columns")
    ap.add_argument("--limit", type=int, default=None,
                    help="only transcribe the next N not-yet-transcribed clips (testing)")
    ap.add_argument("--files", nargs="+", default=None,
                    help="only transcribe these clips (values from the 'file' column)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        sys.exit(f"Not found: {csv_path}")
    base = csv_path.parent
    df = pd.read_csv(csv_path)
    if list(df.columns) != NICK_COLS:
        sys.exit(f"Unexpected columns in {csv_path}: {list(df.columns)}\nExpected: {NICK_COLS}")
    print(f"{len(df)} rows")

    # Load finished transcripts (resume).
    progress_path = base / "asr_progress.jsonl"
    model_tag = f"{ASR_SETTINGS['model']}|{ASR_SETTINGS['compute_type']}"
    done = {}
    if progress_path.exists():
        for line in progress_path.read_text().splitlines():
            rec = json.loads(line)
            if rec["model_tag"] != model_tag:
                sys.exit(f"{progress_path} was made with ASR model '{rec['model_tag']}', but asr.py is now "
                         f"'{model_tag}'. To re-transcribe with the new model, rename the old cache "
                         f"(e.g. to asr_progress_{rec['model_tag'].split('|')[0]}.jsonl) instead of deleting it.")
            if Path(rec["path"]).name != df[PATH_COL].iloc[rec["row"]]:
                sys.exit(f"{progress_path}: row {rec['row']} is {rec['path']}, but clips.csv row "
                         f"{rec['row']} is {df[PATH_COL].iloc[rec['row']]}. Did clips.csv change?")
            done[rec["row"]] = rec
    print(f"{len(done)} clips already transcribed, resuming.")

    # Pick the rows to transcribe now.
    todo = [i for i in range(len(df)) if i not in done]
    if args.files:
        unknown = sorted(set(args.files) - set(df[PATH_COL]))
        if unknown:
            sys.exit(f"Not in the '{PATH_COL}' column of clips.csv: {unknown}")
        todo = [i for i in todo if df[PATH_COL].iloc[i] in set(args.files)]
    if args.limit is not None:
        todo = todo[:args.limit]

    # Check the audio we need exists BEFORE spending time on ASR. A full run needs every file.
    selecting = args.files is not None or args.limit is not None
    check = todo if selecting else list(range(len(df)))
    paths = {i: resolve_path(df[PATH_COL].iloc[i], base) for i in check}
    missing = [df[PATH_COL].iloc[i] for i in check if paths[i] is None]
    if missing:
        sys.exit(f"{len(missing)} of {len(check)} audio files not found in {base / 'clips'}:\n  "
                 + "\n  ".join(missing))
    print(f"All {len(check)} needed audio files found. Transcribing {len(todo)}.")

    start = time.time()
    with progress_path.open("a") as fh:
        for n, i in enumerate(todo, 1):
            rec = {"row": i, "model_tag": model_tag, "path": str(paths[i]), **transcribe_audio(paths[i])}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done[i] = rec
            eta = (time.time() - start) / n * (len(todo) - n) / 60
            print(f"[{n}/{len(todo)}] row {i} {rec['asr_status']} | {rec['transcript'][:60]!r} | ~{eta:.0f} min left")

    if len(done) < len(df):
        print(f"\n{len(done)}/{len(df)} clips transcribed. Run again without --limit/--files to finish.")
        return

    # Score every clip from its transcript and write the deliverables.
    scored = []
    for i in range(len(df)):
        rec = done[i]
        s = score_clip(df[TEXT_COL].iloc[i], rec["transcript"])
        scored.append({**s, **official_columns(s, args.official), "transcript": rec["transcript"],
                       "duration_s": rec["duration_s"], "asr_status": rec["asr_status"]})
    out = df.copy()
    for col in ["wer"] + NEW_COLS:   # fills Nick's empty 'wer' column in place, adds the rest after it
        out[col] = [s[col] for s in scored]
    out = out[NICK_COLS + NEW_COLS]

    out_dir = base / "deliverables"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "clips_arush.csv"
    out.to_csv(out_csv, index=False)

    corpus = {v: corpus_wer(version_rows(scored, v)) for v in VERSIONS}
    capped = {v: int(sum(s[f"wer_{v}"] > 1.0 for s in scored)) for v in VERSIONS}
    capped_cer = {v: int(sum(s[f"cer_{v}"] > 1.0 for s in scored)) for v in VERSIONS}
    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "clips": len(df),
        "asr": {"package": "faster-whisper", "faster_whisper_version": pkg("faster-whisper"),
                "ctranslate2_version": pkg("ctranslate2"), "whisper_model": ASR_SETTINGS["model"],
                "settings": ASR_SETTINGS,
                "rejection": "clips shorter than 0.5 s or with RMS < 1e-3 (about -60 dBFS) are not transcribed"},
        "normalization": {
            "base": f"Whisper EnglishTextNormalizer (whisper-normalizer {pkg('whisper-normalizer')})",
            "pre_rules": [(p, r if isinstance(r, str) else "dotted acronym -> letters joined") for p, r in PRE_RULES],
            "post_rules": [(p, r) for p, r in POST_RULES],
            "order": "lowercase -> PRE_RULES -> Whisper EnglishTextNormalizer -> POST_RULES -> collapse spaces",
            "applied_to": "both reference and transcript"},
        "metrics": {
            "jiwer_version": pkg("jiwer"),
            "wer": "(S + D + I) / N reference words, word-level edit distance after normalization",
            "cer": "character-level edit distance / reference characters (spaces included), after normalization",
            "corpus_cer": "total character errors / total reference characters over the clips in a group",
            "corpus_wer": "total errors / total reference words over all clips (not the mean of per-clip WER)",
            "trailing_dels": "reference words in a deletion chunk at the very end of the jiwer alignment",
            "missing_end_words": "reference words after the last reference word with a correct match (hit); "
                                 "all reference words if there is no hit. Main measure of cut-off endings.",
            "heard_ref_words": "ref_words - missing_end_words: reference words up to the last correctly heard word",
            "heard_errors": "S+D+I between those reference words and the transcript up to the same hit",
            "wer_heard": "heard_errors / heard_ref_words (official version): WER ignoring a cut-off ending; empty when "
                         "no word was heard. Corpus wer_heard = total heard_errors / total heard_ref_words.",
            "capping": "wer and cer = min(official uncapped value, 1.0), rounded to 4 decimals; "
                       "wer_uncapped/cer_uncapped and all *_full/*_no_intro columns are uncapped",
            "clips_capped_wer": capped, "clips_capped_cer": capped_cer},
        "versions": {
            "official": args.official,
            "full": "transcript as is; any intro speech counts as inserted words",
            "intro_phrase_raw": INTRO_PHRASE_RAW,
            "intro_phrase_normalized": " ".join(INTRO_WORDS),
            "no_intro": "align the normalized transcript against (intro phrase words + reference words) with jiwer. "
                        "The intro is the transcript up to and including the last word that EXACTLY matches an "
                        "intro word (earlier substitutions/insertions included; substitutions after it are not, so a "
                        "misheard first word of the sentence stays in the scored part). Deleted intro positions "
                        "are ignored (the intro may be partial or absent). Those words "
                        "(always a leading run) are in intro_text, normalized. The remaining transcript words are "
                        "aligned against the reference alone to give the no_intro wer, cer, subs, dels, ins, "
                        "trailing_dels, missing_end_words and heard_* values."},
        "corpus_wer": {v: corpus[v] for v in VERSIONS},
        "corpus_wer_heard": {"wer": int(out.heard_errors.sum()) / int(out.heard_ref_words.sum()),
                             "errors": int(out.heard_errors.sum()), "heard_ref_words": int(out.heard_ref_words.sum()),
                             "clips_with_nothing_heard": int((out.heard_ref_words == 0).sum())},
        "corpus_cer": corpus_cer_table(df, scored),
        "silent_clips": "clips rejected as silent/short (or with an empty transcript) are kept with "
                        "WER = CER = 1.0 and every reference word counted as a deletion, in both versions",
    }
    (out_dir / "wer_run_info.json").write_text(json.dumps(info, indent=2))

    print(f"\nWrote {out_csv} and {out_dir / 'wer_run_info.json'} (official version: {args.official})")
    for v in VERSIONS:
        print(f"Corpus WER {v}: {corpus[v]['wer']:.4f} ({corpus[v]['errors']} errors / "
              f"{corpus[v]['ref_words']} words) | clips with WER > 1.0: {capped[v]}")
    for v in VERSIONS:
        c = info["corpus_cer"]["all"]["all"][v]
        print(f"Corpus CER {v}: {c['cer']:.4f} ({c['char_errors']} char errors / {c['ref_chars']} chars)")
    print("\nIntro found (clips with / without):")
    for key, g in out.groupby("run_id"):
        has = (g["intro_text"].fillna("") != "").sum()
        print(f"   {key}: {has} / {len(g) - has}")
    odd = [(df[PATH_COL].iloc[i], s["unattributed_lead"]) for i, s in enumerate(scored) if s["unattributed_lead"]]
    print(f"Clips with transcript words before the first reference match that were NOT attributed "
          f"to the intro (possible different intro): {len(odd)}")
    for f, n in odd[:20]:
        print(f"   {f}: {n} words")
    rejected = out[out["asr_status"] != "ok"]
    print(f"\nNot transcribed (scored 1.0): {len(rejected)}")
    for _, r in rejected.iterrows():
        print("   ", r[PATH_COL], "|", r["asr_status"])
    for col in ["run_id", "nfe"]:
        print(f"\nBy {col}:")
        for key, g in out.groupby(col):
            rows = g.to_dict("records")
            print(f"   {key}: " + " | ".join(
                f"{v} {corpus_wer(version_rows(rows, v))['wer']:.4f}, "
                f"missing end words {g[f'missing_end_words_{v}'].mean():.2f}"
                for v in VERSIONS) + f" | clips {len(g)}")


if __name__ == "__main__":
    main()
