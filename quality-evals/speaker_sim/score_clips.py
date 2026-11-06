"""Score speaker similarity for every clip listed in a manifest CSV.

Manifest columns (one row per generated clip):
  run_id      which experiment / config the clip came from (e.g. "sweep")
  prompt_id   which text prompt
  category    short / medium / long (blank -> "all")
  nfe         number of function evaluations used to generate the clip
  seed        generation seed (blank if generation is deterministic)
  wav_path    the generated clip
  ref_path    reference voice clip (blank if there is none)
Relative paths are resolved from the manifest's folder.

Two scores per clip:
  sim_ref     clip vs the reference voice
  sim_hi_nfe  clip vs the same run, prompt and seed at the highest NFE in that group
              (isolates the effect of lowering NFE; 1.0 for the highest-NFE clip itself)

Clips that can't be scored (too short, silent, unreadable) get an error message instead
of a score, so one bad file doesn't stop the run.

Usage:
  python score_clips.py manifest.csv -o scored.csv
"""
import argparse
import csv
import sys
from pathlib import Path

KEY = ("run_id", "prompt_id", "nfe", "seed")
REQUIRED = ("run_id", "prompt_id", "nfe", "wav_path")


def read_manifest(path: str | Path) -> list[dict]:
    path = Path(path)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: no rows")
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    seen = set()
    for i, r in enumerate(rows, start=2):  # line numbers as shown in a spreadsheet
        r["nfe"] = int(r["nfe"])
        r["seed"] = (r.get("seed") or "").strip()
        r["category"] = (r.get("category") or "").strip() or "all"
        for col in ("wav_path", "ref_path"):
            p = (r.get(col) or "").strip()
            r[col] = str((path.parent / p).resolve()) if p else ""
        key = tuple(r[k] for k in KEY)
        if key in seen:
            raise ValueError(f"{path}:{i}: duplicate clip {key}")
        seen.add(key)
    return rows


def hi_nfe_partners(rows: list[dict]) -> dict[int, int]:
    """Row index -> index of the same run/prompt/seed at that group's highest NFE."""
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault((r["run_id"], r["prompt_id"], r["seed"]), []).append(i)
    # A group whose top NFE is below its run's top NFE would quietly use a different
    # anchor than its neighbours, so flag it.
    run_max: dict[str, int] = {}
    for r in rows:
        run_max[r["run_id"]] = max(run_max.get(r["run_id"], 0), r["nfe"])
    partners = {}
    for (run, prompt, seed), idx in groups.items():
        top = max(idx, key=lambda i: rows[i]["nfe"])
        if rows[top]["nfe"] < run_max[run]:
            print(f"warning: {run}/{prompt}/seed={seed or '-'} tops out at NFE {rows[top]['nfe']}, "
                  f"not {run_max[run]}; its sim_hi_nfe uses that lower anchor", file=sys.stderr)
        for i in idx:
            partners[i] = top
    return partners


def score(rows: list[dict], sim) -> list[dict]:
    partners = hi_nfe_partners(rows)
    nfes_per_run: dict[str, set] = {}
    for r in rows:
        nfes_per_run.setdefault(r["run_id"], set()).add(r["nfe"])
    out = []
    for i, r in enumerate(rows):
        row = {**r, "hi_nfe": rows[partners[i]]["nfe"], "duration_s": "", "sim_ref": "", "sim_hi_nfe": "", "error": ""}
        try:
            row["duration_s"] = round(len(sim.load(r["wav_path"])) / 16_000, 3)  # load() returns 16 kHz
            if r["ref_path"]:
                row["sim_ref"] = round(sim.similarity(r["wav_path"], r["ref_path"]), 4)
            # A fixed-NFE run (e.g. a baseline) has nothing to compare against.
            if len(nfes_per_run[r["run_id"]]) > 1:
                j = partners[i]
                row["sim_hi_nfe"] = 1.0 if j == i else round(sim.similarity(r["wav_path"], rows[j]["wav_path"]), 4)
        except Exception as e:  # keep going; the error is recorded in the output row
            row["error"] = f"{type(e).__name__}: {e}"
        out.append(row)
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            print(f"scored {i + 1}/{len(rows)}", file=sys.stderr)
    return out


def write_csv(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest")
    p.add_argument("-o", "--out", default="results/scored.csv")
    args = p.parse_args()

    from speaker_sim import SpeakerSim  # imported here so --help works without loading torch
    rows = score(read_manifest(args.manifest), SpeakerSim())
    write_csv(rows, args.out)
    errors = [r for r in rows if r["error"]]
    print(f"wrote {args.out}: {len(rows)} clips, {len(errors)} errors")
    for r in errors[:10]:
        print(f"  {r['wav_path']}: {r['error']}")


if __name__ == "__main__":
    main()
