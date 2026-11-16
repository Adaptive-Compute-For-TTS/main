"""Wall-clock benchmark: no profiler attached, no monkey patches.

    python bench.py --engine torch-compiled --mode stream --text paragraph --iters 10
"""

import argparse
import json
import statistics
import time

import soundfile as sf
import torch

from harness import add_common_args, gen_kwargs, new_run_dir, reseed, run_once, warmup
from tts import load_engine


def time_stream(engine, **kwargs) -> dict[str, float]:
    start = time.perf_counter()
    ttfa = None
    samples = 0
    for chunk in engine.generate_stream(**kwargs):
        if ttfa is None:
            torch.cuda.synchronize()
            ttfa = time.perf_counter() - start
        samples += int(chunk.shape[-1])
    torch.cuda.synchronize()
    total = time.perf_counter() - start
    audio = samples / engine.sample_rate
    return {"ttfa_s": ttfa or total, "total_s": total, "audio_s": audio, "rtf": total / audio}


def time_generate(engine, **kwargs) -> dict[str, float]:
    start = time.perf_counter()
    samples = int(run_once(engine, "generate", profiled=False, **kwargs).shape[-1])
    total = time.perf_counter() - start
    audio = samples / engine.sample_rate
    return {"total_s": total, "audio_s": audio, "rtf": total / audio}


def summarize(results: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary = {}
    for key in results[0]:
        values = sorted(r[key] for r in results)
        summary[key] = {
            "mean": statistics.fmean(values),
            "p50": statistics.median(values),
            "min": values[0],
            "max": values[-1],
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    engine = load_engine(args.engine)
    warmup(engine, args)

    timer = time_stream if args.mode == "stream" else time_generate
    results = []
    for i in range(args.iters):
        reseed(args)
        results.append(timer(engine, **gen_kwargs(args)))
        print(f"iter {i + 1}/{args.iters}: " + " ".join(f"{k}={v:.4f}" for k, v in results[-1].items()))

    summary = summarize(results)
    print(f"\n{args.engine} / {args.mode} / {args.text} ({args.iters} iters)")
    for key, stats in summary.items():
        print(f"  {key:8s} " + "  ".join(f"{k}={v:.4f}" for k, v in stats.items()))

    run_dir = new_run_dir("bench", args)
    (run_dir / "results.json").write_text(
        json.dumps({"summary": summary, "iters": results}, indent=2)
    )

    # untimed seeded run kept for listening / diffing against other engines
    reseed(args)
    audio = run_once(engine, args.mode, profiled=False, **gen_kwargs(args))
    sf.write(run_dir / "sample.wav", audio.squeeze().float().cpu().numpy(), engine.sample_rate)
    print(f"saved {run_dir}")


if __name__ == "__main__":
    main()
