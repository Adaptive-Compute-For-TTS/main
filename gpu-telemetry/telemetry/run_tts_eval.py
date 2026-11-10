"""Run real dots.tts over benchmark prompts with the telemetry harness attached.

Per prompt: one timed generation, peak VRAM, GPU sampler summary, audio stats.
Saves one run-record JSON per generation plus one summary CSV.
"""

import argparse
import csv
import json
import os
from datetime import datetime

from . import memory
from .dots_model import DEFAULT_MODEL, audio_duration_s, get_runtime, run_dots
from .nvml_sampler import GpuSampler
from .record import EST, OUTPUTS_DIR, RunRecord
from .timing import time_callable

DEFAULT_NUM_STEPS = 10
PROMPT_KEYS = ("id", "text", "length_bucket", "category")
SUMMARY_FIELDS = ["run_id", "prompt_id", "length_bucket", "nfe",
                  "latency_s", "audio_s", "rtf", "peak_vram_mb"]


def load_prompts(path):
    with open(path, encoding="utf-8") as f:
        prompts = json.load(f)
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"{path}: expected a non-empty JSON list of prompts")
    for i, p in enumerate(prompts):
        missing = [k for k in PROMPT_KEYS if k not in p]
        if missing:
            raise ValueError(f"{path}: prompt #{i} is missing {missing}")
    return prompts


def _fmt(value, spec):
    """Format a reading; passes "n/a" (CPU-only placeholders) through untouched."""
    return format(value, spec) if isinstance(value, (int, float)) else str(value)


def run_eval(prompts, num_steps, model=DEFAULT_MODEL, out_dir=OUTPUTS_DIR):
    eval_id = datetime.now(EST).strftime("%Y%m%d_%H%M%S")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, f"tts_eval_{eval_id}.csv")

    get_runtime(model)
    print(f"eval {eval_id}: model={model} nfe={num_steps} prompts={len(prompts)}")
    print(f"warmup on {prompts[0]['id']} (discarded)\n")
    run_dots(prompts[0]["text"], num_steps, model=model)

    print(f"{'prompt_id':<16}  {'bucket':<8}  {'latency s':>9}  {'rtf':>6}  {'peak VRAM MB':>12}")
    print(f"{'-' * 16}  {'-' * 8}  {'-' * 9}  {'-' * 6}  {'-' * 12}")

    # newline="" stops Windows turning csv's \r\n into \r\r\n (blank rows).
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()

        for p in prompts:
            record = RunRecord(config={
                "model": model,
                "nfe": num_steps,
                "prompt_id": p["id"],
                "text": p["text"],
                "length_bucket": p["length_bucket"],
                "category": p["category"],
                "eval_id": eval_id,
            })

            # time_callable discards fn's return value, so capture r here.
            out = {}

            def generate(text=p["text"]):
                out["r"] = run_dots(text, num_steps, model=model)

            memory.reset()
            sampler = GpuSampler()
            timing = time_callable(generate, num_steps, warmups=0, repeats=1, measure=sampler)
            peak = memory.read_peak()
            r = out["r"]

            latency_s = timing["median_ms"] / 1000.0  # repeats=1: the single sample
            audio_s = audio_duration_s(r)
            rtf = float(r["rtf"])
            sample_rate = int(r["sample_rate"])

            record.metrics["timing"] = {**timing, "latency_s": latency_s,
                                        "model_time_used_s": float(r["time_used"])}
            record.metrics["memory"] = peak
            record.metrics["gpu"] = {**sampler.summary(), "interval_ms": sampler.interval_ms}
            record.metrics["audio"] = {"audio_s": audio_s, "rtf": rtf,
                                       "sample_rate": sample_rate}
            record.save(out_dir)

            peak_mb = peak["peak_allocated_mb"]
            writer.writerow({
                "run_id": record.run_id,
                "prompt_id": p["id"],
                "length_bucket": p["length_bucket"],
                "nfe": num_steps,
                "latency_s": latency_s,
                "audio_s": audio_s,
                "rtf": rtf,
                "peak_vram_mb": peak_mb,
            })
            f.flush()  # a crash mid-eval keeps every finished row

            print(f"{str(p['id']):<16}  {str(p['length_bucket']):<8}  {latency_s:>9.2f}  "
                  f"{rtf:>6.3f}  {_fmt(peak_mb, '>12.0f')}")

    return summary_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True,
                        help="JSON list of {id, text, length_bucket, category}")
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS,
                        help=f"ODE steps / NFE per generation (default: {DEFAULT_NUM_STEPS})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"dots.tts checkpoint (default: {DEFAULT_MODEL})")
    parser.add_argument("--out-dir", default=OUTPUTS_DIR,
                        help="where run-records and the CSV go (default: outputs/)")
    args = parser.parse_args()
    if args.num_steps < 1:
        parser.error("need --num-steps >= 1")

    summary_path = run_eval(load_prompts(args.prompts), args.num_steps,
                            args.model, args.out_dir)
    print(f"\nsummary: {summary_path}")


if __name__ == "__main__":
    main()
