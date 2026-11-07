"""Checks for the Phase 2 pipeline (score_clips.py, analyze.py) on tiny hand-made tables.

Runs in seconds and doesn't load the WavLM model: a fake scorer stands in for it.
Exits non-zero if any check fails.
"""
import csv
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

import analyze
import build_run_json as brj
import score_clips

failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))


def raises(fn) -> str:
    try:
        fn()
    except ValueError as e:
        return str(e)
    return ""


def write(path: Path, rows: list[dict]) -> Path:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


class FakeSim:
    """Stands in for SpeakerSim: SIM is read from the file name, e.g. clip_0.8.wav -> 0.8."""

    def load(self, path):
        if "silent" in path:
            raise ValueError(f"{path}: (near-)silent audio")
        return [0.0] * 16_000 * 2  # 2 seconds

    def similarity(self, a, b):
        num = lambda p: float(Path(p).stem.split("_")[-1])
        return round(num(a) * num(b), 4)


tmp = Path(tempfile.mkdtemp())

# ---------- score_clips ----------
manifest = write(tmp / "manifest.csv", [
    dict(run_id="sweep", prompt_id="p1", category="short", nfe=n, seed=0, wav_path=f"p1_{n}_{v}.wav", ref_path="ref_1.0.wav")
    for n, v in [(4, 0.5), (8, 0.8), (32, 1.0)]
] + [
    dict(run_id="sweep", prompt_id="p2", category="long", nfe=n, seed=0, wav_path=f"p2_{n}_{v}.wav", ref_path="")
    for n, v in [(4, 0.6), (8, 0.9)]  # p2 has no NFE 32 clip
] + [
    dict(run_id="base", prompt_id="p1", category="short", nfe=16, seed=s, wav_path=f"b_{s}_{v}.wav", ref_path="ref_1.0.wav")
    for s, v in [(0, 0.9), (1, 0.7)]
] + [
    dict(run_id="base", prompt_id="p3", category="", nfe=16, seed=0, wav_path="silent_1.wav", ref_path="ref_1.0.wav")
])
rows = score_clips.read_manifest(manifest)
check("manifest: blank category becomes 'all'", rows[-1]["category"] == "all")
check("manifest: relative paths resolved from the manifest's folder", rows[0]["wav_path"] == str(tmp.resolve() / "p1_4_0.5.wav"))

dup = write(tmp / "dup.csv", [dict(run_id="r", prompt_id="p", nfe=4, seed=0, wav_path="a.wav")] * 2)
check("manifest: duplicate clip is rejected", "duplicate" in raises(lambda: score_clips.read_manifest(dup)))
bad = write(tmp / "bad.csv", [dict(run_id="r", prompt_id="p", wav_path="a.wav")])
check("manifest: missing nfe column is rejected", "missing columns" in raises(lambda: score_clips.read_manifest(bad)))

scored = {(r["run_id"], r["prompt_id"], r["nfe"], r["seed"]): r for r in score_clips.score(rows, FakeSim())}
s = scored[("sweep", "p1", 4, "0")]
check("sim_ref = clip vs reference", s["sim_ref"] == 0.5, str(s["sim_ref"]))
check("sim_hi_nfe = clip vs same prompt+seed at highest NFE", s["sim_hi_nfe"] == 0.5 and s["hi_nfe"] == 32,
      f"{s['sim_hi_nfe']} vs NFE {s['hi_nfe']}")
check("highest-NFE clip scores 1.0 against itself", scored[("sweep", "p1", 32, "0")]["sim_hi_nfe"] == 1.0)
s = scored[("sweep", "p2", 4, "0")]
check("group missing the top NFE falls back to its own top (and warns)", s["hi_nfe"] == 8 and s["sim_hi_nfe"] == 0.54,
      f"NFE {s['hi_nfe']}, {s['sim_hi_nfe']}")
check("no reference -> sim_ref left blank", s["sim_ref"] == "")
check("fixed-NFE run gets no sim_hi_nfe", scored[("base", "p1", 16, "0")]["sim_hi_nfe"] == "")
s = scored[("base", "p3", 16, "0")]
check("unscorable clip records an error instead of crashing", "silent" in s["error"] and s["sim_ref"] == "", s["error"])
check("duration recorded", scored[("sweep", "p1", 4, "0")]["duration_s"] == 2.0)
score_clips.write_csv(list(scored.values()), tmp / "scored.csv")

# ---------- joining ----------
clips = analyze.build_clips(tmp / "scored.csv")
full_key = write(tmp / "wer_key.csv", [dict(run_id=c["run_id"], prompt_id=c["prompt_id"], nfe=c["nfe"], seed=c["seed"],
                                            wer=0.1 * i) for i, c in enumerate(clips)])
analyze.join_metric(clips, full_key, {"wer": "wer"}, False)
check("join by run/prompt/nfe/seed", [round(c["wer"], 6) for c in clips] == [round(0.1 * i, 6) for i in range(len(clips))])

by_file = write(tmp / "wer_file.csv", [dict(filename=f"some/dir/{Path(c['wav_path']).name}", WER=0.5) for c in clips])
analyze.join_metric(clips, by_file, {"wer": "WER"}, False)
check("join by filename, custom column name", all(c["wer"] == 0.5 for c in clips))

per_nfe = write(tmp / "lat_nfe.csv", [dict(nfe=n, latency_ms=100 + 10 * n) for n in (4, 8, 16, 32)])
analyze.join_metric(clips, per_nfe, {"latency_ms": "latency_ms", "rtf": "rtf"}, False)
check("join one-number-per-NFE latency (rtf column optional)", all(c["latency_ms"] == 100 + 10 * c["nfe"] for c in clips))

dupe = write(tmp / "dupe.csv", [dict(nfe=4, wer=0.1), dict(nfe=4, wer=0.2)])
check("duplicate metric rows are rejected", "duplicate" in raises(lambda: analyze.join_metric(clips, dupe, {"wer": "wer"}, False)))
partial = write(tmp / "partial.csv", [dict(nfe=4, wer=0.3)])
check("unmatched clips stop the run", "no match" in raises(lambda: analyze.join_metric(clips, partial, {"wer": "wer"}, False)))
fresh = analyze.build_clips(tmp / "scored.csv")
analyze.join_metric(fresh, partial, {"wer": "wer"}, True)
check("--allow-missing leaves unmatched clips blank",
      all((c["wer"] == 0.3) if c["nfe"] == 4 else math.isnan(c["wer"]) for c in fresh))
nocol = write(tmp / "nocol.csv", [dict(nfe=4, word_error=0.3)])
check("missing metric column is reported", "missing column" in raises(lambda: analyze.join_metric(clips, nocol, {"wer": "wer"}, False)))
nokey = write(tmp / "nokey.csv", [dict(speaker="x", wer=0.3)])
check("file with no key or filename column is rejected", "needs some of" in raises(lambda: analyze.join_metric(clips, nokey, {"wer": "wer"}, False)))

arush = write(tmp / "arush.csv", [dict(file=Path(c["wav_path"]).name, run_id=c["run_id"], prompt_id=c["prompt_id"],
                                       nfe=c["nfe"], seed=c["seed"], wer=0.5, cer=0.4, wer_heard=0.1, missing_end_words=i % 2)
                                  for i, c in enumerate(analyze.build_clips(tmp / "scored.csv"))])
ac = analyze.build_clips(tmp / "scored.csv", wer=arush)
check("Arush-style file: cer, wer_heard and truncation picked up",
      all(c["cer"] == 0.4 and c["wer_heard"] == 0.1 and c["truncated"] == float(i % 2) for i, c in enumerate(ac)))

lat_clips = analyze.build_clips(tmp / "scored.csv", latency=per_nfe)
c = next(c for c in lat_clips if c["nfe"] == 8)
check("RTF computed from latency / audio length when not given", abs(c["rtf"] - 0.18 / 2.0) < 1e-9, f"{c['rtf']:.4f}")

# ---------- statistics ----------
n, mean, lo, hi = analyze.bootstrap_mean([1, 2, 3, 4, 5, 6, 7, 8], list("abcdefgh"))
check("bootstrap: mean right, CI brackets it", mean == 4.5 and lo < 4.5 < hi, f"{mean} [{lo:.2f}, {hi:.2f}]")
check("bootstrap: same answer every run", analyze.bootstrap_mean([1, 2, 3, 4], list("abcd")) == analyze.bootstrap_mean([1, 2, 3, 4], list("abcd")))
n, mean, lo, hi = analyze.bootstrap_mean([1, float("nan"), 3], list("abc"))
check("bootstrap: blanks ignored", n == 2 and mean == 2.0)
n, mean, lo, hi = analyze.bootstrap_mean([1, 3], ["a", "a"])
check("bootstrap: one prompt -> no CI (not a fake narrow one)", mean == 2 and math.isnan(lo))
# Two prompts, many identical clips each: resampling clips would give a falsely tiny CI.
n, mean, lo, hi = analyze.bootstrap_mean([0.0] * 50 + [1.0] * 50, ["a"] * 50 + ["b"] * 50)
check("bootstrap resamples prompts, not clips", lo == 0.0 and hi == 1.0, f"[{lo}, {hi}]")

a, b, r2 = analyze.fit_line([4, 8, 16, 32], [180 + 25 * x for x in (4, 8, 16, 32)])
check("latency fit recovers overhead and per-NFE cost", abs(a - 180) < 1e-6 and abs(b - 25) < 1e-6 and abs(r2 - 1) < 1e-9,
      f"{a:.1f} ms + {b:.1f} ms/NFE")

toy = [dict(run_id="r", prompt_id=p, category="all", seed="0", nfe=n, hi_nfe=32,
            sim_ref=v, sim_hi_nfe=math.nan, wer=w, latency_ms=100 + n, rtf=math.nan, duration_s=1.0)
       for p, n, v, w in [("p1", 4, 0.70, 0.30), ("p1", 32, 0.90, 0.05), ("p2", 4, 0.80, 0.20), ("p2", 32, 0.95, 0.04)]]
d = {(x["nfe"], x["metric"]): x for x in analyze.paired_deltas(toy)}
check("paired delta: SIM change vs own prompt at top NFE", abs(d[(4, "sim_ref")]["mean_delta"] - (-0.175)) < 1e-9,
      f"{d[(4, 'sim_ref')]['mean_delta']:.3f}")
check("paired delta: WER change", abs(d[(4, "wer")]["mean_delta"] - 0.205) < 1e-9)
check("paired delta: no row for the top NFE itself", (32, "sim_ref") not in d)
fits = analyze.latency_fits(toy)
check("latency_fits runs per run/category", len(fits) == 1 and abs(fits[0]["per_nfe"] - 1) < 1e-9)
check("failed share counts unscorable clips", next(s for s in analyze.summarize(analyze.build_clips(tmp / "scored.csv"))
      if s["run_id"] == "base" and s["metric"] == "failed" and s["category"] == "all")["mean"] == 1 / 3)
summ = analyze.summarize(toy)
check("summary has a row per NFE and metric", {(s["nfe"], s["metric"]) for s in summ} >=
      {(4, "sim_ref"), (32, "sim_ref"), (4, "wer"), (32, "latency_ms")})


# ---------- build_run_json: invalid-clip flags and per-run JSON ----------
import json
import re

src = Path(__file__).with_name("speaker_sim.py").read_text()
k = brj.scorer_constants()
check("flag rules are read from speaker_sim.py itself",
      k["MIN_SECONDS"] == float(re.search(r"^MIN_SECONDS = ([\d.]+)", src, re.M).group(1))
      and k["MIN_PEAK"] == float(re.search(r"^MIN_PEAK = ([\d.]+)", src, re.M).group(1))
      and re.search(r'^MODEL_REVISION = "(\w+)"', src, re.M).group(1) == k["MODEL_REVISION"])

wavs = tmp / "run_a"
wavs.mkdir()
(wavs / "run_metadata.json").write_text(json.dumps({"max_generate_length": 10, "seed": 0}))  # cap = 10 x 0.16 = 1.6 s
t = lambda secs: np.arange(int(16_000 * secs)) / 16_000
sf.write(wavs / "good.wav", 0.3 * np.sin(2 * np.pi * 220 * t(1.0)), 16_000)
sf.write(wavs / "quiet.wav", 0.002 * np.sin(2 * np.pi * 220 * t(1.0)), 16_000)
sf.write(wavs / "tiny.wav", 0.3 * np.sin(2 * np.pi * 220 * t(0.2)), 16_000)
sf.write(wavs / "capped.wav", 0.3 * np.sin(2 * np.pi * 220 * t(1.6)), 16_000)
(wavs / "broken.wav").write_text("this is not audio")
check("good clip has no audio flag", brj.audio_flags(wavs / "good.wav", k) == [])
check("silent clip flagged", brj.audio_flags(wavs / "quiet.wav", k) == ["silent"])
check("clip under 0.5 s flagged", brj.audio_flags(wavs / "tiny.wav", k) == ["too_short"])
check("corrupt file flagged", brj.audio_flags(wavs / "broken.wav", k) == ["unreadable_audio"])
check("missing file flagged", brj.audio_flags(wavs / "nope.wav", k) == ["missing_audio"])


def clip(name, prompt="p1", nfe=4, cat="short", wer=0.0, rtf=1.0, dur=1.0, run="run_a", error="", hi=32, sim=0.9):
    return dict(run_id=run, prompt_id=prompt, category=cat, content="plain", nfe=nfe, seed="0", wav_path=str(wavs / name),
                hi_nfe=hi, duration_s=dur, sim_ref=sim, sim_hi_nfe=math.nan, wer=wer, wer_heard=math.nan, cer=wer,
                missing_end_words=0.0, latency_ms=rtf * dur * 1000, rtf=rtf, error=error)


def flagged(clips, team=None, asr=None):
    brj.flag_clips(clips, k, team, asr)
    return [(c["valid"], c["flags"]) for c in clips]


check("all-clear clip is valid with no flags", flagged([clip("good.wav")]) == [(True, [])])
check("silent clip is invalid; the scorer's error adds nothing",
      flagged([clip("quiet.wav", error="ValueError: quiet.wav: (near-)silent audio", sim=math.nan)]) == [(False, ["silent"])])
check("scorer error with no audio explanation -> score_error",
      flagged([clip("good.wav", error="RuntimeError: boom", sim=math.nan)]) == [(False, ["score_error"])])
check("clip at the generation cap -> truncated_at_cap",
      flagged([clip("capped.wav", dur=1.6)]) == [(False, ["truncated_at_cap"])])
check("clip one patch (0.16 s) short of the cap is not truncated", flagged([clip("good.wav", dur=1.44)]) == [(True, [])])
team = {("run_a", "p1", 4): dict(status="contaminated_continuation", signal_ok="False"),
        ("run_a", "p2", 4): dict(status="clean_candidate", signal_ok="False"),
        ("run_a", "p3", 4): dict(status="clean_candidate", signal_ok="True")}
check("team's contaminated flag -> invalid",
      flagged([clip("good.wav", prompt="p1")], team=team) == [(False, ["contaminated_continuation"])])
check("failed signal check is a caution: still valid",
      flagged([clip("good.wav", prompt="p2")], team=team) == [(True, ["signal_check_failed"])])
check("cautions are not added to clips that are already invalid",
      flagged([clip("quiet.wav", prompt="p2")], team=team) == [(False, ["silent"])])
asr = {("run_a", "p1", 4): "rejected: silent (RMS 0.00017 < 0.001)", ("run_a", "p2", 4): "ok"}
check("ASR rejection -> invalid", flagged([clip("good.wav", prompt="p1")], asr=asr) == [(False, ["asr_rejected"])])
check("clip with a missing metric -> missing_metric", flagged([clip("good.wav", wer=math.nan)]) == [(False, ["missing_metric"])])
check("a missing metric adds nothing to a clip already invalid",
      flagged([clip("quiet.wav", sim=math.nan)]) == [(False, ["silent"])])
group = [clip("good.wav", prompt=f"p{i}", rtf=1.0) for i in range(3)] + [clip("good.wav", prompt="p9", rtf=4.0)]
res = flagged(group)
check("latency outlier (4x the group median) -> caution, still valid",
      res[3] == (True, ["latency_outlier"]) and all(r == (True, []) for r in res[:3]))
check("outlier rule needs a group of 3+",
      flagged([clip("good.wav", rtf=1.0), clip("good.wav", prompt="p2", rtf=9.0)]) == [(True, [])] * 2)
check("outliers are judged within their own NFE",
      all(r == (True, []) for r in flagged([clip("good.wav", prompt=f"p{i}", nfe=4, rtf=1.0) for i in range(3)]
                                           + [clip("good.wav", prompt="p9", nfe=32, rtf=4.0)])))

# End to end on a tiny run: 2 prompts x NFE 4/32; p2 at NFE 32 is silent
rows = [clip("good.wav", prompt="p1", nfe=4, wer=0.1, sim=0.80), clip("good.wav", prompt="p1", nfe=32, wer=0.0, sim=0.90),
        clip("good.wav", prompt="p2", nfe=4, wer=0.3, sim=0.70),
        clip("quiet.wav", prompt="p2", nfe=32, sim=math.nan, error="ValueError: x: (near-)silent audio")]
brj.flag_clips(rows, k, None, None)
doc = brj.build_run("run_a", rows, k, brj.flag_table(k), False, False)
doc = json.loads(json.dumps(doc, allow_nan=False))  # strictly valid JSON
v = doc["validity"]
check("validity counts add up", v["n_clips"] == 4 and v["n_valid"] == 3 and v["n_invalid"] == 1 and v["invalid_by_flag"] == {"silent": 1})
check("invalid clips stay in clips, marked", [c["valid"] for c in doc["clips"]] == [True, True, True, False]
      and doc["clips"][3]["flags"] == ["silent"])
sim4 = next(s for s in doc["summary_by_nfe"] if s["category"] == "all" and s["nfe"] == 4 and s["metric"] == "sim_ref")
check("aggregates cover valid clips only", sim4["n"] == 2 and abs(sim4["mean"] - 0.75) < 1e-9)
check("clip with an invalid top-NFE partner gets no paired delta",
      {(d["nfe"], d["metric"]): d["n"] for d in doc["paired_deltas"]}.get((4, "wer")) == 1)
check("every flag used is defined", {f for c in doc["clips"] for f in c["flags"]} <= set(doc["flag_definitions"]))
check("run with no valid clips builds, with empty aggregates",
      not (lambda r: r["summary_by_nfe"] or r["paired_deltas"] or r["latency_fit"])(
          brj.build_run("run_a", [dict(c, flags=["silent"], valid=False) for c in rows], k, brj.flag_table(k), False, False)))
frontend_ok = lambda c: (isinstance(c["seed"], str) and all(isinstance(c[f], (int, float)) for f in ("wer", "cer", "latency_ms", "rtf"))
                         and (c["sim_ref"] is None or isinstance(c["sim_ref"], float)) and isinstance(c["error"], str))
check("clips keep the types the frontend's parseExperimentClips requires", all(frontend_ok(c) for c in doc["clips"]))
check("error text carries no local folder paths",
      "/" not in brj.clip_record(dict(clip("good.wav", error=f"ValueError: {wavs}/quiet.wav: (near-)silent audio"), valid=False, flags=[]))["error"])

print(f"\n{'all checks passed' if not failures else f'{failures} check(s) FAILED'}")
sys.exit(1 if failures else 0)
