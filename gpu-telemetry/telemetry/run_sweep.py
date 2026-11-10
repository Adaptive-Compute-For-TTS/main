"""Sweep NFE values; time each, measure peak memory and sample GPU stats, save a run-record per NFE.

With --profile, each NFE also gets one torch.profiler pass (kernel launches, op counts).
"""

import argparse
import csv
import os
from datetime import datetime
from functools import partial

from . import memory
from .nvml_sampler import DEFAULT_INTERVAL_MS, GpuSampler, summary_keys
from .profiler import profile_callable
from .record import EST, OUTPUTS_DIR, RunRecord
from .timing import time_callable
from .toy_model import run_toy_model

DEFAULT_NFES = [4, 8, 16, 32, 64]
DEFAULT_WARMUPS = 3
DEFAULT_REPEATS = 10
# Long enough for NVML (whose utilization window is up to 1 s) to see the runs.
DEFAULT_MIN_DURATION_S = 1.0
SUMMARY_FIELDS = [
    "nfe", "median_ms", "per_step_ms", "min_ms", "max_ms", "repeats", "measured_s",
    "peak_allocated_mb", "peak_reserved_mb",
    *summary_keys(), "gpu_num_samples",
    "run_id",
]
# Scalar fields of metrics.profile; added to the CSV (before run_id) only with --profile.
PROFILE_FIELDS = ["kernel_launches", "distinct_kernels", "gpu_kernel_time_ms", "cpu_op_calls"]


def _fmt(value):
    """Format a reading for the console; passes "n/a" through untouched."""
    return f"{value:.1f}" if isinstance(value, float) else str(value)


def run_sweep(nfes, warmups=DEFAULT_WARMUPS, repeats=DEFAULT_REPEATS,
              min_duration_s=DEFAULT_MIN_DURATION_S, sample_interval_ms=DEFAULT_INTERVAL_MS,
              profile=False, out_dir=OUTPUTS_DIR):
    """Measure the toy model at each NFE; save one run-record each plus a CSV summary.

    With `profile`, also run one profiler pass per NFE, after the measured runs.
    Returns (rows, summary_path).
    """
    sweep_id = datetime.now(EST).strftime("%Y%m%d_%H%M%S")
    print(f"sweep {sweep_id}: nfe={nfes} warmups={warmups} min_repeats={repeats} "
          f"min_duration_s={min_duration_s} sample_interval_ms={sample_interval_ms} "
          f"profile={profile}\n")
    header = (f"{'NFE':>5}  {'median ms':>10}  {'per-step ms':>11}  {'repeats':>7}  "
              f"{'peak alloc MB':>13}  {'util %':>6}  {'power W':>7}  {'samples':>7}")
    rule = (f"{'-' * 5}  {'-' * 10}  {'-' * 11}  {'-' * 7}  "
            f"{'-' * 13}  {'-' * 6}  {'-' * 7}  {'-' * 7}")
    if profile:
        header += f"  {'kernels':>7}  {'cpu ops':>7}"
        rule += f"  {'-' * 7}  {'-' * 7}"
    print(header)
    print(rule)

    rows = []
    for nfe in nfes:
        record = RunRecord(config={"model": "toy", "nfe": nfe, "sweep_id": sweep_id})
        # Peak memory covers warm-ups and repeats; they are identical runs, so
        # it equals the peak of any single one. The sampler covers only the
        # timed repeats, so warm-up (clock ramp, one-time init) can't skew means.
        memory.reset()
        sampler = GpuSampler(interval_ms=sample_interval_ms)
        timing = time_callable(partial(run_toy_model, nfe), nfe, warmups, repeats,
                               measure=sampler, min_duration_s=min_duration_s)
        peak = memory.read_peak()
        gpu = {**sampler.summary(), "interval_ms": sampler.interval_ms}
        record.metrics["timing"] = timing
        record.metrics["memory"] = peak
        record.metrics["gpu"] = gpu
        if profile:
            # A separate pass after timing and the memory read, so profiler
            # overhead and allocations touch neither. The model is already warm.
            record.metrics["profile"] = profile_callable(partial(run_toy_model, nfe), nfe,
                                                         warmups=0)
        record.save(out_dir)

        row = {
            "nfe": nfe,
            "median_ms": timing["median_ms"],
            "per_step_ms": timing["per_step_ms"],
            "min_ms": timing["min_ms"],
            "max_ms": timing["max_ms"],
            "repeats": timing["repeats"],
            "measured_s": timing["measured_s"],
            **peak,
            **{key: gpu[key] for key in summary_keys()},
            "gpu_num_samples": gpu["num_samples"],
            "run_id": record.run_id,
        }
        line = (f"{nfe:>5}  {timing['median_ms']:>10.2f}  {timing['per_step_ms']:>11.3f}  "
                f"{timing['repeats']:>7}  {_fmt(peak['peak_allocated_mb']):>13}  "
                f"{_fmt(gpu['util_mean_pct']):>6}  {_fmt(gpu['power_mean_w']):>7}  "
                f"{gpu['num_samples']:>7}")
        if profile:
            prof = record.metrics["profile"]
            row.update({key: prof[key] for key in PROFILE_FIELDS})
            line += f"  {prof['kernel_launches']:>7}  {prof['cpu_op_calls']:>7}"
        rows.append(row)
        print(line)

    # Profile columns go just before run_id, so run_id stays last either way.
    fields = SUMMARY_FIELDS[:-1] + PROFILE_FIELDS + ["run_id"] if profile else SUMMARY_FIELDS
    summary_path = os.path.join(out_dir, f"sweep_{sweep_id}.csv")
    # newline="" stops Windows turning csv's \r\n into \r\r\n (blank rows).
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows, summary_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nfe", type=int, nargs="+", default=DEFAULT_NFES,
                        help=f"NFE values to sweep (default: {DEFAULT_NFES})")
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help=f"minimum timed repeats per NFE (default: {DEFAULT_REPEATS})")
    parser.add_argument("--min-duration-s", type=float, default=DEFAULT_MIN_DURATION_S,
                        help="keep repeating until the timed section lasts this long "
                             f"(default: {DEFAULT_MIN_DURATION_S}; 0 disables)")
    parser.add_argument("--sample-interval-ms", type=float, default=DEFAULT_INTERVAL_MS,
                        help=f"GPU sampling period in ms (default: {DEFAULT_INTERVAL_MS})")
    parser.add_argument("--profile", action="store_true",
                        help="also run one torch.profiler pass per NFE (kernel launches, "
                             "op counts); off by default because it is slow")
    args = parser.parse_args()
    if min(args.nfe) < 1 or args.warmups < 0 or args.repeats < 1:
        parser.error("need every --nfe >= 1, --warmups >= 0, --repeats >= 1")
    if args.min_duration_s < 0 or args.sample_interval_ms <= 0:
        parser.error("need --min-duration-s >= 0, --sample-interval-ms > 0")

    _, summary_path = run_sweep(args.nfe, args.warmups, args.repeats,
                                args.min_duration_s, args.sample_interval_ms, args.profile)
    print(f"\nsummary: {summary_path}")


if __name__ == "__main__":
    main()
