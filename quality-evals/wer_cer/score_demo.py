"""Demo task #9: per-WAV WER/CER as one small JSON keyed run_id__prompt_id__nfe.

One WAV (e.g. right after a live generation):
  .venv/bin/python score_demo.py --wav out.wav --text "The kettle is already on the stove." \
      --run-id run_123 --prompt-id p001 --nfe 20 --out wer_cer.json

Many WAVs (manifest CSV with columns audio_path, text, prompt_id, nfe, and optionally run_id;
audio_path is relative to the manifest's folder, or absolute):
  .venv/bin/python score_demo.py --manifest runs/demo_clean_set/manifest.csv --run-id xvector_4090 \
      --out runs/demo_clean_set/wer_cer_demo_small.json --limit 3

Another Whisper model (same settings otherwise; use a separate --out, the script refuses to mix models):
  ... --asr-model large-v3 --out runs/demo_clean_set/wer_cer_demo_large-v3.json

The output JSON is updated in place: new clips are added and an existing key is replaced, so many runs can
share one file. Transcripts are cached next to it in transcripts_<model>.jsonl (keyed by the WAV's SHA-256),
so re-running never re-transcribes the same audio. Scores come from score_clip() (official version no_intro)
and are uncapped. Corpus numbers are total errors / total reference words over every clip in the file.
Silent or too-short clips are kept with WER = CER = 1.0 and valid = false.

Exit codes: 0 = ok, 2 = bad input.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from asr import ASR_SETTINGS, set_model
from wer_scoring import pkg, score_clip

EVAL_NAME = "wer_cer"
EVAL_VERSION = "1.0"
OFFICIAL = "no_intro"
REQUIRED = ["audio_path", "text", "prompt_id", "nfe"]


def fail(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(2)


def clip_key(run_id, prompt_id, nfe):
    return f"{run_id}__{prompt_id}__nfe{int(nfe)}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def num(x, nd=4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


def score_entry(item, asr):
    """One clip's JSON entry from its request (run_id, prompt_id, nfe, text, audio) and ASR output."""
    s = score_clip(item["text"], asr["transcript"])
    o = OFFICIAL
    valid = asr["asr_status"] == "ok"
    flags = [] if valid else [asr["asr_status"]]
    if s["intro_text"]:
        flags.append("intro_detected")  # leaked voice-prompt intro; set aside by no_intro scoring
    return {
        "run_id": item["run_id"], "prompt_id": item["prompt_id"], "nfe": int(item["nfe"]),
        "audio": item["audio"], "audio_sha256": asr["audio_sha256"],
        "reference_text": item["text"], "transcript": asr["transcript"],
        "asr_status": asr["asr_status"], "valid": valid, "flags": flags,
        "duration_s": asr["duration_s"],
        "wer": num(s[f"wer_{o}"]), "cer": num(s[f"cer_{o}"]),
        "word_errors": s[f"subs_{o}"] + s[f"dels_{o}"] + s[f"ins_{o}"], "ref_words": s["ref_words"],
        "subs": s[f"subs_{o}"], "dels": s[f"dels_{o}"], "ins": s[f"ins_{o}"],
        "char_errors": s[f"char_errors_{o}"], "ref_chars": s[f"ref_chars_{o}"],
        "missing_end_words": s[f"missing_end_words_{o}"], "intro_text": s["intro_text"],
    }


def corpus(entries):
    """Corpus WER/CER = total errors / total reference words (or characters). Rejected clips count (WER 1.0)."""
    words = sum(e["ref_words"] for e in entries)
    chars = sum(e["ref_chars"] for e in entries)
    w_err = sum(e["word_errors"] for e in entries)
    c_err = sum(e["char_errors"] for e in entries)
    return {"clips": len(entries), "valid_clips": sum(e["valid"] for e in entries),
            "wer": num(w_err / words) if words else None, "cer": num(c_err / chars) if chars else None,
            "word_errors": w_err, "ref_words": words, "char_errors": c_err, "ref_chars": chars}


def summarize(clips):
    """The whole output file: header, corpus totals, totals by NFE, and every clip sorted by key."""
    entries = list(clips.values())
    by_nfe = {}
    for e in entries:
        by_nfe.setdefault(e["nfe"], []).append(e)
    return {
        "eval": EVAL_NAME, "eval_version": EVAL_VERSION, "official_version": OFFICIAL,
        "key": "run_id__prompt_id__nfe",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "asr": dict(ASR_SETTINGS),
        "packages": {p: pkg(p) for p in ("faster-whisper", "ctranslate2", "jiwer", "whisper-normalizer")},
        "corpus": corpus(entries),
        "by_nfe": {str(k): corpus(v) for k, v in sorted(by_nfe.items())},
        "clips": dict(sorted(clips.items())),
    }


def load_requests(args):
    """Return the list of clips to score: dicts with key, run_id, prompt_id, nfe, text, audio, path."""
    if args.wav:
        for flag in ("text", "run_id", "prompt_id", "nfe"):
            if getattr(args, flag) is None:
                fail(f"--wav needs --{flag.replace('_', '-')}")
        rows, base = [{"audio_path": args.wav, "text": args.text, "prompt_id": args.prompt_id,
                       "nfe": args.nfe}], Path.cwd()
    else:
        manifest = Path(args.manifest)
        if not manifest.is_file():
            fail(f"manifest not found: {manifest}")
        df = pd.read_csv(manifest, dtype={"prompt_id": str, "run_id": str})
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            fail(f"manifest is missing columns: {missing}")
        rows, base = df.to_dict("records"), manifest.resolve().parent

    reqs, seen = [], set()
    for i, r in enumerate(rows, 1):
        run_id = r.get("run_id") if isinstance(r.get("run_id"), str) and r["run_id"].strip() else args.run_id
        if not run_id:
            fail(f"row {i}: no run_id (add a run_id column or pass --run-id)")
        text = r["text"] if isinstance(r["text"], str) else ""
        if not text.strip():
            fail(f"row {i}: empty text")
        try:
            nfe = int(r["nfe"])
        except (TypeError, ValueError):
            fail(f"row {i}: nfe {r['nfe']!r} is not an integer")
        p = Path(str(r["audio_path"]))
        p = p if p.is_absolute() else base / p
        if not p.is_file():
            fail(f"row {i}: audio not found: {p}")
        key = clip_key(run_id, r["prompt_id"], nfe)
        if key in seen:
            fail(f"row {i}: duplicate key {key}")
        seen.add(key)
        reqs.append({"key": key, "run_id": run_id, "prompt_id": str(r["prompt_id"]), "nfe": nfe,
                     "text": text, "audio": str(r["audio_path"]), "path": p})
    return reqs


def load_cache(path):
    cache = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["asr"] != ASR_SETTINGS:
                    fail(f"{path} has transcripts from different ASR settings: {r['asr']}")
                cache[r["audio_sha256"]] = r
    return cache


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)  # atomic: a reader never sees a half-written file


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--wav", help="score one WAV (needs --text, --run-id, --prompt-id, --nfe)")
    src.add_argument("--manifest", help="CSV with audio_path, text, prompt_id, nfe [, run_id]")
    ap.add_argument("--text", help="the sentence the WAV should say")
    ap.add_argument("--run-id", help="run id (manifest: used for rows without a run_id)")
    ap.add_argument("--prompt-id")
    ap.add_argument("--nfe")
    ap.add_argument("--out", required=True, help="output JSON, created or updated in place")
    ap.add_argument("--limit", type=int, help="score only the first N clips (for testing)")
    ap.add_argument("--asr-model", default=ASR_SETTINGS["model"],
                    help=f"Whisper model (default {ASR_SETTINGS['model']}); other ASR settings are unchanged")
    args = ap.parse_args()
    set_model(args.asr_model)

    reqs = load_requests(args)
    if args.limit is not None:
        reqs = reqs[:args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    clips = {}
    if out.is_file():
        old = json.loads(out.read_text())
        if old.get("asr") != ASR_SETTINGS:
            fail(f"{out} was made with different ASR settings {old.get('asr')}; use a new --out file")
        clips = old["clips"]

    cache_path = out.parent / f"transcripts_{ASR_SETTINGS['model']}.jsonl"
    cache = load_cache(cache_path)
    for i, r in enumerate(reqs, 1):
        digest = sha256(r["path"])
        asr = cache.get(digest)
        if asr is None:
            from wer_scoring import transcribe_audio  # loads Whisper on first use
            asr = {"audio_sha256": digest, **transcribe_audio(r["path"]), "asr": dict(ASR_SETTINGS)}
            with open(cache_path, "a") as fh:
                fh.write(json.dumps(asr, ensure_ascii=False) + "\n")
            cache[digest] = asr
            how = "transcribed"
        else:
            how = "cached"
        e = score_entry(r, asr)
        clips[r["key"]] = e
        print(f"[{i}/{len(reqs)}] {r['key']}: WER {e['wer']} CER {e['cer']} "
              f"({e['word_errors']}/{e['ref_words']} words, {e['asr_status']}, {how})", flush=True)

    data = summarize(clips)
    write_json(out, data)
    c = data["corpus"]
    print(f"\n{c['clips']} clips in {out} ({c['clips'] - c['valid_clips']} invalid) | "
          f"corpus WER {c['wer']} ({c['word_errors']}/{c['ref_words']})  CER {c['cer']} | ASR {ASR_SETTINGS['model']}")


if __name__ == "__main__":
    main()
