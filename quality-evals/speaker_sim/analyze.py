"""Join SIM scores with WER and latency, average per NFE with 95% CIs, and plot.

Inputs:
  scored.csv     output of score_clips.py (one row per clip)
  --wer FILE     CSV with a `wer` column (rename with --wer-col); `cer`, `wer_heard` and
                 `missing_end_words` are picked up too if present (Arush's format has them)
  --latency FILE CSV with a `latency_ms` column (rename with --latency-col), optional `rtf`

How WER/latency rows are matched to clips: by whichever of run_id, prompt_id, nfe, seed
the file has. So a latency file with only `nfe, latency_ms` (one number per NFE) works too.
If the file has none of those, it is matched by audio filename (a `wav`, `file`, `filename`,
`wav_path` or `path` column). Any clip without a match stops the run (use --allow-missing
to leave it blank instead); duplicate matches always stop it.

Outputs in --out (default results/):
  clips.csv            one row per clip: SIM, WER, latency, RTF
  summary_by_nfe.csv   mean + 95% CI per run / category / NFE / metric
  paired_deltas.csv    change vs the same prompt+seed at the highest NFE, mean + 95% CI
  latency_fit.csv      latency_ms (and RTF) = intercept + per_nfe * NFE, with 95% CIs
  summary.json         all of the above, for the frontend
  plots/*.png

CIs are bootstrap percentiles that resample *prompts* (with all their clips), because clips
of the same prompt at different seeds/NFEs are not independent.

Usage:
  python analyze.py results/scored.csv --wer wer.csv --latency latency.csv
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

KEY = ("run_id", "prompt_id", "nfe", "seed")
FILE_COLS = ("wav", "file", "filename", "wav_path", "path")
# failed = 1 for a clip that couldn't be scored (e.g. a silent generation), so failures count
# against their NFE instead of silently dropping out of the averages.
# wer_heard = WER on only the words the clip got to before it was cut off (separates "said it
# wrong" from "ran out of time"); truncated = 1 if words are missing at the end.
METRICS = ("sim_ref", "sim_hi_nfe", "wer", "wer_heard", "cer", "truncated", "latency_ms", "rtf", "failed")
DELTA_METRICS = ("sim_ref", "wer", "wer_heard", "latency_ms", "rtf")  # sim_hi_nfe is already relative
OPTIONAL = {"rtf", "cer", "wer_heard", "missing_end_words"}  # used if the file has them
ALL = "all"
N_BOOT = 10_000


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def _norm(col: str, v: str) -> str:
    v = (v or "").strip()
    return str(int(float(v))) if col == "nfe" and v else v


def read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------- joining ----------

def join_metric(clips: list[dict], path: str | Path, cols: dict[str, str], allow_missing: bool) -> None:
    """Copy columns from a metric CSV into the matching clip rows (in place).

    cols maps the name in `clips` -> the column name in the file. A file column that
    doesn't exist is skipped if optional (rtf), otherwise it's an error.
    """
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"{path}: no rows")
    header = rows[0].keys()
    cols = {ours: theirs for ours, theirs in cols.items() if theirs in header or ours not in OPTIONAL}
    absent = [theirs for theirs in cols.values() if theirs not in header]
    if absent:
        raise ValueError(f"{path}: missing column(s) {absent}; columns are {list(header)}")

    key_cols = [k for k in KEY if k in header]
    if key_cols:
        clip_key = lambda c: tuple(_norm(k, str(c[k])) for k in key_cols)
        row_key = lambda r: tuple(_norm(k, r[k]) for k in key_cols)
        how = ", ".join(key_cols)
    else:
        fcol = next((c for c in FILE_COLS if c in header), None)
        if fcol is None:
            raise ValueError(f"{path}: needs some of {KEY} or a filename column {FILE_COLS} to match clips")
        names = [Path(c["wav_path"]).name for c in clips]
        if len(set(names)) != len(names):
            raise ValueError(f"{path}: clip filenames aren't unique, so can't match by filename")
        clip_key = lambda c: (Path(c["wav_path"]).name,)
        row_key = lambda r: (Path(r[fcol].strip()).name,)
        how = f"filename ({fcol})"

    table = {}
    for i, r in enumerate(rows, start=2):
        k = row_key(r)
        if k in table:
            raise ValueError(f"{path}:{i}: duplicate entry for {dict(zip(key_cols or ['file'], k))}")
        table[k] = r

    used, missing = set(), []
    for c in clips:
        k = clip_key(c)
        r = table.get(k)
        if r is None:
            missing.append(k)
            continue
        used.add(k)
        for ours, theirs in cols.items():
            c[ours] = _num(r[theirs])
    if missing:
        msg = f"{path}: {len(missing)} clip(s) have no match (by {how}), e.g. {missing[:3]}"
        if not allow_missing:
            raise ValueError(msg + "; use --allow-missing to leave them blank")
        print("warning: " + msg, file=sys.stderr)
    if len(used) < len(table):
        print(f"warning: {path}: {len(table) - len(used)} row(s) matched no clip", file=sys.stderr)
    print(f"joined {path} by {how}: {len(clips) - len(missing)}/{len(clips)} clips", file=sys.stderr)


def build_clips(scored_path, wer=None, latency=None, wer_col="wer", latency_col="latency_ms",
                rtf_col="rtf", allow_missing=False) -> list[dict]:
    clips = read_csv(scored_path)
    for c in clips:
        c["nfe"] = int(c["nfe"])
        c["hi_nfe"] = int(c["hi_nfe"])
        for m in ("sim_ref", "sim_hi_nfe", "duration_s"):
            c[m] = _num(c[m])
        c.update(wer=math.nan, wer_heard=math.nan, cer=math.nan, missing_end_words=math.nan, truncated=math.nan,
                 latency_ms=math.nan, rtf=math.nan, failed=1.0 if c["error"] else 0.0)
    if wer:
        join_metric(clips, wer, {"wer": wer_col, "cer": "cer", "wer_heard": "wer_heard",
                                 "missing_end_words": "missing_end_words"}, allow_missing)
        for c in clips:
            if not math.isnan(c["missing_end_words"]):
                c["truncated"] = float(c["missing_end_words"] > 0)
    if latency:
        join_metric(clips, latency, {"latency_ms": latency_col, "rtf": rtf_col}, allow_missing)
        for c in clips:  # RTF = generation time / audio length, if the file didn't give it
            if math.isnan(c["rtf"]) and c["duration_s"] > 0:
                c["rtf"] = c["latency_ms"] / 1000 / c["duration_s"]
    return clips


# ---------- statistics ----------

def bootstrap_mean(values, clusters, n_boot=N_BOOT, seed=0) -> tuple[int, float, float, float]:
    """(n, mean, ci_low, ci_high): 95% percentile bootstrap that resamples whole clusters."""
    v = np.asarray(values, dtype=float)
    keep = ~np.isnan(v)
    v, cl = v[keep], np.asarray(clusters)[keep]
    if v.size == 0:
        return 0, math.nan, math.nan, math.nan
    ids, inv = np.unique(cl, return_inverse=True)
    if ids.size < 2:
        return int(v.size), float(v.mean()), math.nan, math.nan
    sums = np.bincount(inv, weights=v)
    counts = np.bincount(inv).astype(float)
    pick = np.random.default_rng(seed).integers(0, ids.size, size=(n_boot, ids.size))
    means = sums[pick].sum(1) / counts[pick].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return int(v.size), float(v.mean()), float(lo), float(hi)


def fit_line(x, y) -> tuple[float, float, float]:
    """Least squares y = a + b*x -> (a, b, r2)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    b, a = np.polyfit(x, y, 1)
    ss_res = float(((y - (a + b * x)) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(a), float(b), (1 - ss_res / ss_tot) if ss_tot > 0 else math.nan


def _groups(clips):
    """Yield (run_id, category, rows) for each run, overall and per category."""
    for run in sorted({c["run_id"] for c in clips}):
        rows = [c for c in clips if c["run_id"] == run]
        yield run, ALL, rows
        cats = sorted({c["category"] for c in rows})
        if cats != [ALL]:
            for cat in cats:
                yield run, cat, [c for c in rows if c["category"] == cat]


def summarize(clips) -> list[dict]:
    out = []
    for run, cat, rows in _groups(clips):
        for nfe in sorted({c["nfe"] for c in rows}):
            sel = [c for c in rows if c["nfe"] == nfe]
            for m in METRICS:
                n, mean, lo, hi = bootstrap_mean([c.get(m, math.nan) for c in sel], [c["prompt_id"] for c in sel])
                if n:
                    out.append(dict(run_id=run, category=cat, nfe=nfe, metric=m, n=n, mean=mean, ci_low=lo, ci_high=hi))
    return out


def paired_deltas(clips) -> list[dict]:
    """Each clip minus its own prompt+seed at the highest NFE, averaged per NFE."""
    top = {(c["run_id"], c["prompt_id"], c["seed"]): c for c in clips if c["nfe"] == c["hi_nfe"]}
    out = []
    for run, cat, rows in _groups(clips):
        for nfe in sorted({c["nfe"] for c in rows if c["nfe"] < c["hi_nfe"]}):
            sel = [c for c in rows if c["nfe"] == nfe]
            for m in DELTA_METRICS:
                d = [c.get(m, math.nan) - top[(c["run_id"], c["prompt_id"], c["seed"])].get(m, math.nan) for c in sel]
                n, mean, lo, hi = bootstrap_mean(d, [c["prompt_id"] for c in sel])
                if n:
                    out.append(dict(run_id=run, category=cat, nfe=nfe, vs_nfe=sel[0]["hi_nfe"], metric=m, n=n,
                                    mean_delta=mean, ci_low=lo, ci_high=hi,
                                    ci_excludes_zero=bool(lo > 0 or hi < 0) if not math.isnan(lo) else ""))
    return out


def latency_fits(clips, metrics=("latency_ms", "rtf"), n_boot=2_000, seed=0) -> list[dict]:
    """metric = intercept + per_nfe * NFE, CI by resampling prompts.

    For latency_ms the intercept is the fixed overhead in ms. For RTF (seconds of compute per
    second of audio) it's the fixed cost per second of audio, which is fairer when clip lengths differ.
    """
    out = []
    rng = np.random.default_rng(seed)
    for run, cat, all_rows in _groups(clips):
        for m in metrics:
            rows = [c for c in all_rows if not math.isnan(c[m])]
            if len({c["nfe"] for c in rows}) < 2:
                continue
            a, b, r2 = fit_line([c["nfe"] for c in rows], [c[m] for c in rows])
            by_prompt: dict[str, list] = {}
            for c in rows:
                by_prompt.setdefault(c["prompt_id"], []).append((c["nfe"], c[m]))
            prompts = list(by_prompt)
            boots = []
            for _ in range(n_boot if len(prompts) > 1 else 0):
                pts = [p for i in rng.integers(0, len(prompts), len(prompts)) for p in by_prompt[prompts[i]]]
                xs, ys = zip(*pts)
                if len(set(xs)) > 1:
                    boots.append(fit_line(xs, ys)[:2])
            (a_lo, b_lo), (a_hi, b_hi) = (np.percentile(boots, [2.5, 97.5], axis=0) if boots
                                          else ((math.nan,) * 2, (math.nan,) * 2))
            out.append(dict(run_id=run, category=cat, metric=m, n=len(rows), intercept=a,
                            intercept_ci_low=float(a_lo), intercept_ci_high=float(a_hi), per_nfe=b,
                            per_nfe_ci_low=float(b_lo), per_nfe_ci_high=float(b_hi), r2=r2))
    return out


# ---------- output ----------

def _clean(v):
    if isinstance(v, float):
        return "" if math.isnan(v) else round(v, 6)
    return v


def write_rows(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: _clean(v) for k, v in r.items()} for r in rows)


CLIP_COLS = ("run_id", "prompt_id", "category", "nfe", "seed", "sim_ref", "sim_hi_nfe", "wer", "wer_heard",
             "cer", "missing_end_words", "latency_ms", "rtf", "duration_s", "hi_nfe", "wav", "error")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scored", help="output of score_clips.py")
    p.add_argument("--wer")
    p.add_argument("--latency")
    p.add_argument("--wer-col", default="wer")
    p.add_argument("--latency-col", default="latency_ms")
    p.add_argument("--rtf-col", default="rtf")
    p.add_argument("--allow-missing", action="store_true", help="leave unmatched clips blank instead of stopping")
    p.add_argument("--category-col", default="category",
                   help="manifest column to break results down by (e.g. 'content' instead of short/medium/long)")
    p.add_argument("--no-plots", action="store_true", help="tables only (e.g. for breakdowns with many categories)")
    p.add_argument("-o", "--out", default="results")
    args = p.parse_args()

    clips = build_clips(args.scored, args.wer, args.latency, args.wer_col, args.latency_col,
                        args.rtf_col, args.allow_missing)
    if args.category_col != "category":
        for c in clips:
            c["category"] = c[args.category_col] or "all"
    errors = [c for c in clips if c["error"]]
    if errors:
        print(f"warning: {len(errors)} clip(s) had scoring errors and have no SIM", file=sys.stderr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = [{k: (Path(c["wav_path"]).name if k == "wav" else c[k]) for k in CLIP_COLS} for c in clips]
    for row in table:  # shared outputs shouldn't carry anyone's local folder paths
        row["error"] = re.sub(r"(?:/[^/\s:]+)+/([^/\s:]+)", r"\1", row["error"])
    summary, deltas, fits = summarize(clips), paired_deltas(clips), latency_fits(clips)
    write_rows(table, out / "clips.csv")
    write_rows(summary, out / "summary_by_nfe.csv")
    write_rows(deltas, out / "paired_deltas.csv")
    write_rows(fits, out / "latency_fit.csv")
    everything = {"clips": table, "summary_by_nfe": summary, "paired_deltas": deltas, "latency_fit": fits}
    with open(out / "summary.json", "w") as f:
        json.dump(_jsonable(everything), f, indent=1, allow_nan=False)

    if args.no_plots:
        print(f"wrote {out}/: clips.csv ({len(table)} rows), summary_by_nfe.csv, paired_deltas.csv, "
              f"latency_fit.csv, summary.json")
        return
    import plots  # matplotlib is only needed for this step
    swept = sorted({c["run_id"] for c in clips if c["nfe"] != c["hi_nfe"]})  # runs with more than one NFE
    made = [p for run in swept for p in plots.make_all(clips, summary, fits, run, out / "plots" / run)]
    if len(swept) > 1:
        made += plots.compare_runs(summary, swept, out / "plots" / "compare_runs")
    print(f"wrote {out}/: clips.csv ({len(table)} rows), summary_by_nfe.csv, paired_deltas.csv, "
          f"latency_fit.csv, summary.json, {len(made)} plots")


def _jsonable(x):
    """NaN isn't valid JSON; the frontend gets null instead."""
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


if __name__ == "__main__":
    main()
