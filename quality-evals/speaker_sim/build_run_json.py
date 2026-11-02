"""One JSON per run for the dashboard: SIM, WER/CER and runtime joined per clip, invalid clips flagged.

A clip is *invalid* when it can't support an NFE-vs-quality comparison: nothing to hear, the wrong
words, cut off by the generation cap, or a metric is missing. Invalid clips stay in `clips` with
`valid: false` and their reasons in `flags`, and are left out of the aggregates. *Cautions* (a latency
outlier, a failed signal check) are listed in `flags` too but don't change `valid`, and are only
attached to clips that are otherwise valid.

Inputs (all produced by earlier steps):
  scored.csv      score_clips.py output: SIM, audio length and the path of each WAV
  --wer FILE      per-clip ASR output (wer_arush.csv)
  --latency FILE  latency/RTF per clip (prepare_watai_exp.py)
  --flags FILE    the team's experiments/clean_samples/wav_flags.csv (contaminated / truncated clips),
                  or --no-team-flags to skip it (contamination then goes undetected)

Output in --out: <run_id>.json for each run, index.json, and with --plot <run_id>_nfe_quality.png.
`clips` has the fields of the frontend's ExperimentClip plus `content`, `valid` and `flags`;
`summary_by_nfe`, `paired_deltas` and `latency_fit` have the same rows as analyze.py's summary.json
but cover valid clips only (that file, which covers all clips, is unchanged).

Usage:
  python build_run_json.py results/watai_exp/scored.csv --wer results/watai_exp/wer_arush.csv \\
      --latency results/watai_exp/latency.csv --flags /path/to/watai_exp/experiments/clean_samples/wav_flags.csv \\
      -o results/watai_exp/runs --plot
"""
import argparse
import ast
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

import analyze

SCHEMA_VERSION = 1
PATCH_SECONDS = 0.16  # audio per generated patch (watai_exp clean_samples/run_summary.json)
CAP_TOLERANCE_S = 0.02
OUTLIER_FACTOR = 3.0  # latency_outlier: RTF above 3x the median of its run / NFE / length bucket
REQUIRED = ("sim_ref", "wer", "cer", "latency_ms", "rtf")
CLIP_FIELDS = ("run_id", "prompt_id", "category", "content", "nfe", "seed", "sim_ref", "sim_hi_nfe", "wer",
               "wer_heard", "cer", "missing_end_words", "latency_ms", "rtf", "duration_s", "hi_nfe", "wav", "error")
CONFIG_KEYS = ("model", "voice_conditioning", "precision", "seed", "ode_method", "guidance_scale",
               "speaker_scale", "max_generate_length")
# GPU labels come from the run owners; run_metadata.json only says "cuda".
GPU = {"xvector_4090": "RTX 4090", "orig_3090": "RTX 3090"}
WER_SOURCE = "Whisper small via tts_eval_org (Arush); WER with any re-spoken reference intro removed, capped at 1"
NFE_DEFINITION = "Euler vector-field evaluations per generated audio patch, one per step (experiments/nfe_benchmark.py)"


def scorer_constants(path: Path = Path(__file__).with_name("speaker_sim.py")) -> dict:
    """MODEL_ID, MODEL_REVISION, MIN_SECONDS and MIN_PEAK read from the scorer's source, so the flags
    use exactly the scorer's own rules (importing it would pull in torch)."""
    want = {"MODEL_ID", "MODEL_REVISION", "MIN_SECONDS", "MIN_PEAK"}
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in want:
            out[node.targets[0].id] = ast.literal_eval(node.value)
    if set(out) != want:
        raise ValueError(f"{path}: couldn't read {sorted(want - set(out))}")
    return out


def flag_table(k: dict) -> dict[str, tuple[str, str]]:
    """flag -> (severity, description). Order is the order flags are listed on a clip."""
    return {
        "missing_audio": ("invalid", "The WAV file is not on disk."),
        "unreadable_audio": ("invalid", "The WAV file could not be decoded."),
        "too_short": ("invalid", f"Shorter than {k['MIN_SECONDS']} s: too little audio for a speaker embedding."),
        "silent": ("invalid", f"(Near-)silent: peak below {k['MIN_PEAK']} (about -40 dBFS) or RMS below 1e-4. "
                              "A failed generation, not a quiet voice."),
        "asr_rejected": ("invalid", "The ASR step rejected the clip, so its WER and CER are not a real measurement."),
        "score_error": ("invalid", "The SIM scorer failed on this clip."),
        "contaminated_continuation": ("invalid", "Generated in continuation mode: the reference transcript "
                                                 "(\"Hey, Adam here...\") leaks into the output."),
        "truncated_at_cap": ("invalid", "Stopped at the generation length cap (max_generate_length), "
                                        "usually mid-sentence, so WER measures the cap, not NFE."),
        "missing_metric": ("invalid", "SIM, WER, CER, latency or RTF is missing."),
        "latency_outlier": ("caution", f"RTF is over {OUTLIER_FACTOR:g}x the median for its run, NFE and length "
                                       "bucket (warm-up or a noisy machine). Kept in the latency numbers."),
        "signal_check_failed": ("caution", "Failed the team's signal check (clipping, long leading silence, a "
                                           "pause over 1 s, or an abrupt end). Worth listening to."),
    }


def audio_flags(path: Path, k: dict) -> list[str]:
    """Check the WAV itself, with the scorer's own rules (too short first, then silent)."""
    if not path.exists():
        return ["missing_audio"]
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:  # soundfile raises its own error types for corrupt or non-audio files
        return ["unreadable_audio"]
    mono = data.mean(axis=1)
    if mono.size < sr * k["MIN_SECONDS"]:
        return ["too_short"]
    if math.sqrt(float(np.mean(mono**2))) < 1e-4 or float(np.abs(mono).max()) < k["MIN_PEAK"]:
        return ["silent"]
    return []


def read_team_flags(path: Path) -> dict[tuple[str, str, int], dict]:
    """wav_flags.csv rows keyed by (run folder, prompt_id, NFE); first repeat only."""
    rows = {}
    for r in analyze.read_csv(path):
        if r["repeat"] != "1":
            continue
        key = (r["run"], r["prompt_id"], int(r["NFE"]))
        if key in rows:
            raise ValueError(f"{path}: duplicate entry for {key}")
        rows[key] = r
    return rows


def read_asr_status(path: Path) -> dict[tuple[str, str, int], str]:
    return {(r["run_id"], r["prompt_id"], int(r["nfe"])): r["asr_status"] for r in analyze.read_csv(path)}


def cap_seconds(folder: Path) -> float | None:
    """Length cap of the run folder (max_generate_length patches), if its run_metadata.json is there."""
    meta = folder / "run_metadata.json"
    return json.loads(meta.read_text())["max_generate_length"] * PATCH_SECONDS if meta.exists() else None


def flag_clips(clips: list[dict], k: dict, team: dict | None, asr: dict | None) -> dict:
    """Set c["flags"] and c["valid"] on every clip; returns counts about how well the sources agreed."""
    caps: dict[Path, float | None] = {}
    report = Counter()
    for c in clips:
        wav = Path(c["wav_path"])
        flags = audio_flags(wav, k)
        if c["error"] and not flags:
            flags.append("score_error")
        if asr is not None:
            status = asr.get((c["run_id"], c["prompt_id"], c["nfe"]))
            if status is None:
                report["no_asr_row"] += 1
            elif status != "ok":
                flags.append("asr_rejected")
        team_row = None
        if team is not None:
            team_row = team.get((wav.parent.name, c["prompt_id"], c["nfe"]))
            if team_row is None:
                report["no_team_row"] += 1
            elif team_row["status"] in ("contaminated_continuation", "truncated_at_cap"):
                flags.append(team_row["status"])
        if wav.parent not in caps:
            caps[wav.parent] = cap_seconds(wav.parent)
        cap = caps[wav.parent]
        at_cap = cap is not None and c["duration_s"] >= cap - CAP_TOLERANCE_S
        if at_cap and "truncated_at_cap" not in flags:
            flags.append("truncated_at_cap")
            if team_row is not None and team_row["status"] != "contaminated_continuation":
                report["cap_not_in_team_flags"] += 1
        if (team_row is not None and team_row["status"] == "truncated_at_cap") and cap is not None and not at_cap:
            report["team_cap_not_derived"] += 1
        if not flags and any(math.isnan(c[m]) for m in REQUIRED):
            flags.append("missing_metric")
        c["flags"] = flags
        c["_signal_ok"] = None if team_row is None else team_row["signal_ok"] == "True"
    order = list(flag_table(k))
    outliers = latency_outliers([c for c in clips if not c["flags"]])
    for c in clips:
        if not c["flags"]:
            if id(c) in outliers:
                c["flags"].append("latency_outlier")
            if c["_signal_ok"] is False:
                c["flags"].append("signal_check_failed")
        c["valid"] = not any(flag_table(k)[f][0] == "invalid" for f in c["flags"])
        c["flags"].sort(key=order.index)
    return dict(report)


def latency_outliers(clips: list[dict]) -> set[int]:
    """ids of clips whose RTF is over OUTLIER_FACTOR x the median of their run / NFE / length bucket."""
    groups: dict[tuple, list[dict]] = {}
    for c in clips:
        if not math.isnan(c["rtf"]):
            groups.setdefault((c["run_id"], c["nfe"], c["category"]), []).append(c)
    out = set()
    for rows in groups.values():
        if len(rows) >= 3:
            med = statistics.median(c["rtf"] for c in rows)
            out |= {id(c) for c in rows if c["rtf"] > OUTLIER_FACTOR * med}
    return out


def with_partner(clips: list[dict]) -> list[dict]:
    """Only clips whose own prompt+seed at the highest NFE is also in the list, so paired deltas stay paired."""
    tops = {(c["run_id"], c["prompt_id"], c["seed"]) for c in clips if c["nfe"] == c["hi_nfe"]}
    return [c for c in clips if (c["run_id"], c["prompt_id"], c["seed"]) in tops]


def run_config(clips: list[dict], run_id: str) -> dict:
    """Generation settings from the run folders' run_metadata.json / prompts.csv; every folder must agree."""
    folders = sorted({Path(c["wav_path"]).parent for c in clips})
    metas = [json.loads((d / "run_metadata.json").read_text()) for d in folders if (d / "run_metadata.json").exists()]
    cfg = {"seed": int(clips[0]["seed"])}
    if metas:
        first = {key: metas[0][key] for key in CONFIG_KEYS if key in metas[0]}
        for m in metas[1:]:
            if {key: m[key] for key in CONFIG_KEYS if key in m} != first:
                raise ValueError(f"{run_id}: run folders disagree on their settings")
        cfg.update(first)
        cfg["cap_seconds"] = round(first["max_generate_length"] * PATCH_SECONDS, 3)
    refs = {Path(r["reference_audio"]).name for d in folders if (d / "prompts.csv").exists()
            for r in analyze.read_csv(d / "prompts.csv") if r.get("reference_audio")}
    if len(refs) == 1:
        cfg["reference_audio"] = refs.pop()
    if run_id in GPU:
        cfg["gpu"] = GPU[run_id]
        cfg["gpu_source"] = "run owner; run_metadata.json only records 'cuda'"
    return cfg


def _round(x, digits=6):
    if isinstance(x, dict):
        return {key: _round(v, digits) for key, v in x.items()}
    if isinstance(x, list):
        return [_round(v, digits) for v in x]
    return round(x, digits) if isinstance(x, float) else x


def clip_record(c: dict) -> dict:
    rec = {f: Path(c["wav_path"]).name if f == "wav" else c[f] for f in CLIP_FIELDS}
    rec["error"] = re.sub(r"(?:/[^/\s:]+)+/([^/\s:]+)", r"\1", c["error"])  # no local folder paths in shared files
    rec["valid"], rec["flags"] = c["valid"], c["flags"]
    return rec


def build_run(run_id: str, clips: list[dict], k: dict, table: dict, team_used: bool, asr_used: bool) -> dict:
    clips = sorted(clips, key=lambda c: (c["prompt_id"], c["nfe"]))
    valid = [c for c in clips if c["valid"]]
    invalid = [c for c in clips if not c["valid"]]
    sev = lambda c, kind: [f for f in c["flags"] if table[f][0] == kind]
    summary = analyze.summarize(valid)
    deltas = analyze.paired_deltas(with_partner(valid))
    fits = analyze.latency_fits(valid) if valid else []
    doc = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "aggregates_cover": "valid_clips_only",
        "config": run_config(clips, run_id),
        "nfe_definition": NFE_DEFINITION,
        "metrics": {
            "sim": {"model": k["MODEL_ID"], "revision": k["MODEL_REVISION"],
                    "sim_ref": "cosine similarity of the clip to the reference voice it was cloned from",
                    "sim_hi_nfe": "cosine similarity to the same prompt and seed at the highest NFE (hi_nfe)"},
            "wer": WER_SOURCE,
            "latency": "generation time per clip (latency_ms); rtf = latency / audio length",
        },
        "validity": {
            "n_clips": len(clips),
            "n_valid": len(valid),
            "n_invalid": len(invalid),
            "invalid_by_flag": dict(Counter(f for c in invalid for f in sev(c, "invalid"))),
            "caution_by_flag": dict(Counter(f for c in clips for f in sev(c, "caution"))),
            "valid_by_nfe": {str(n): sum(c["nfe"] == n for c in valid) for n in sorted({c["nfe"] for c in clips})},
            "valid_by_category": {cat: sum(c["category"] == cat for c in valid) for cat in sorted({c["category"] for c in clips})},
            "checked_against_team_flags": team_used,
            "checked_against_asr_status": asr_used,
        },
        "flag_definitions": {f: {"severity": s, "description": d} for f, (s, d) in table.items()},
        "clips": [clip_record(c) for c in clips],
        "summary_by_nfe": summary,
        "paired_deltas": deltas,
        "latency_fit": fits,
    }
    return _round(analyze._jsonable(doc))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scored", help="output of score_clips.py")
    p.add_argument("--wer", required=True)
    p.add_argument("--latency", required=True)
    p.add_argument("--flags", help="the team's wav_flags.csv")
    p.add_argument("--no-team-flags", action="store_true", help="skip wav_flags.csv (contaminated clips go undetected)")
    p.add_argument("--plot", action="store_true", help="also write <run_id>_nfe_quality.png for each swept run")
    p.add_argument("-o", "--out", default="results/runs")
    args = p.parse_args()
    if not args.flags and not args.no_team_flags:
        p.error("give --flags wav_flags.csv, or --no-team-flags to skip it")

    k = scorer_constants()
    table = flag_table(k)
    clips = analyze.build_clips(args.scored, args.wer, args.latency)
    team = read_team_flags(Path(args.flags)) if args.flags else None
    asr = read_asr_status(Path(args.wer)) if "asr_status" in analyze.read_csv(args.wer)[0] else None
    report = flag_clips(clips, k, team, asr)
    if report:
        print(f"note: {report}", file=sys.stderr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for run_id in sorted({c["run_id"] for c in clips}):
        run_clips = [c for c in clips if c["run_id"] == run_id]
        doc = build_run(run_id, run_clips, k, table, team is not None, asr is not None)
        path = out / f"{run_id}.json"
        with open(path, "w") as f:
            json.dump(doc, f, indent=1, allow_nan=False)
        v = doc["validity"]
        index.append({"run_id": run_id, "file": path.name, "n_clips": v["n_clips"], "n_valid": v["n_valid"],
                      "n_invalid": v["n_invalid"]})
        print(f"{run_id}: {v['n_valid']}/{v['n_clips']} valid; invalid by flag {v['invalid_by_flag']}; "
              f"cautions {v['caution_by_flag']}")
        if args.plot and len({c["nfe"] for c in run_clips if c["valid"]}) > 1:
            import plots  # matplotlib is only needed here
            note = (f"valid clips only: {v['n_valid']} of {v['n_clips']} "
                    f"(excluded: {', '.join(f'{n} {f}' for f, n in v['invalid_by_flag'].items()) or 'none'})")
            plots.nfe_vs(doc["summary_by_nfe"], run_id, ["sim_ref", "wer", "rtf"],
                         out / f"{run_id}_nfe_quality.png", "NFE vs voice similarity, WER and cost", note=note)
    with open(out / "index.json", "w") as f:
        json.dump({"schema_version": SCHEMA_VERSION, "scorer": {"model": k["MODEL_ID"], "revision": k["MODEL_REVISION"]},
                   "runs": index}, f, indent=1)
    print(f"wrote {len(index)} run file(s) + index.json to {out}/")


if __name__ == "__main__":
    main()
