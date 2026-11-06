"""Turn the NFE sweeps in the team's watai_exp repo into inputs for score_clips.py / analyze.py.

Reads each run's results.csv, prompts.csv and run_metadata.json, checks they agree, and writes:
  manifest.csv   one row per clip (category = short/medium/long, content = plain/numbers/...)
  latency.csv    latency_ms, RTF and audio length per clip, from results.csv

Paths in the repo's CSVs point at the cluster (/home/...), so clips and the reference voice are
looked up by filename inside the local clone.

Usage:
  python prepare_watai_exp.py /path/to/watai_exp -o results/watai_exp
"""
import argparse
import csv
import json
from pathlib import Path

# run_id -> folders under experiments/nfe_benchmark_runs (each holds 27 of the 54 prompts)
RUNS = {
    "orig_3090": ["nfe_54_prompts_v1_full_part_0", "nfe_54_prompts_v1_full_part_1"],
    "xvector_4090": ["nfe_54_prompts_v1_full_xvector_part_0", "nfe_54_prompts_v1_full_xvector_part_1"],
}
PROMPT_SET = "experiments/benchmark_prompts_v1.csv"


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo", type=Path, help="local clone of Adaptive-Compute-For-TTS/watai_exp")
    p.add_argument("-o", "--out", type=Path, default=Path("results/watai_exp"))
    args = p.parse_args()
    repo = args.repo.expanduser().resolve()
    prompt_set = {r["id"]: r for r in read_csv(repo / PROMPT_SET)}

    manifest, latency = [], []
    for run_id, parts in RUNS.items():
        for part in parts:
            d = repo / "experiments/nfe_benchmark_runs" / part
            meta = json.loads((d / "run_metadata.json").read_text())
            prompts = {r["prompt_id"]: r for r in read_csv(d / "prompts.csv")}
            results = read_csv(d / "results.csv")
            expected = len(meta["prompt_ids"]) * len(meta["nfe_values"]) * meta["measured_repeats_per_prompt_nfe"]
            assert len(results) == expected, f"{part}: {len(results)} rows, expected {expected}"
            for r in results:
                pid, nfe = r["prompt_id"], int(r["NFE"])
                pr = prompts[pid]
                assert pr["text"] == prompt_set[pid]["text"], f"{part}/{pid}: prompt text differs from {PROMPT_SET}"
                assert int(r["seed"]) == meta["seed"] and r["repeat"] == "1", f"{part}/{pid}/NFE {nfe}: unexpected seed/repeat"
                wav = d / Path(r["output_path"]).name
                ref = repo / "experiments/my_files" / Path(pr["reference_audio"]).name
                for f in (wav, ref):
                    assert f.exists(), f"missing {f}"
                manifest.append(dict(run_id=run_id, prompt_id=pid, category=prompt_set[pid]["length_bucket"],
                                     content=prompt_set[pid]["category"], nfe=nfe, seed=r["seed"],
                                     wav_path=str(wav), ref_path=str(ref)))
                latency.append(dict(run_id=run_id, prompt_id=pid, nfe=nfe, seed=r["seed"],
                                    latency_ms=round(float(r["latency_seconds"]) * 1000, 1), rtf=r["RTF"],
                                    audio_duration_s=r["audio_duration_seconds"]))

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(manifest, args.out / "manifest.csv")
    write_csv(latency, args.out / "latency.csv")
    for run_id in RUNS:
        n = sum(m["run_id"] == run_id for m in manifest)
        print(f"{run_id}: {n} clips")
    print(f"wrote {args.out}/manifest.csv and latency.csv")


if __name__ == "__main__":
    main()
