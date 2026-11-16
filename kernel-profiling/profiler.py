"""Profile one instrumented run of an engine.

Not named profile.py on purpose: torch._dynamo imports cProfile, which does
`import profile`, and a local profile.py would shadow the stdlib module.

    # torch profiler -> runs/profile/<run>/trace.json (open in perfetto)
    python profiler.py --engine torch-compiled --mode stream --tool torch

    # nsys / ncu: only the measured run is captured (cudaProfilerStart/Stop)
    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
        -o runs/profile/nsys_stream python profiler.py --mode stream --tool nsys
    ncu --profile-from-start off --set full -k regex:dgrad \
        -o runs/profile/ncu_stream python profiler.py --mode stream --tool ncu
"""

import argparse
import contextlib

import torch
from torch.profiler import ProfilerActivity, profile

from harness import add_common_args, gen_kwargs, new_run_dir, reseed, run_once, warmup
from tts import load_engine


def profile_torch(engine, args) -> None:
    reseed(args)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        run_once(engine, args.mode, profiled=True, **gen_kwargs(args))

    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=args.rows)
    print(table)

    run_dir = new_run_dir("profile", args)
    (run_dir / "table.txt").write_text(table)
    prof.export_chrome_trace(str(run_dir / "trace.json"))
    print(f"saved {run_dir}")


def profile_external(engine, args) -> None:
    # emit_nvtx turns ops and record_function ranges into NVTX ranges that
    # nsys shows on the timeline and ncu can filter on (--nvtx-include)
    reseed(args)
    with torch.cuda.profiler.profile(), torch.autograd.profiler.emit_nvtx():
        run_once(engine, args.mode, profiled=True, **gen_kwargs(args))

    # the report itself goes wherever nsys/ncu -o points; this keeps the context
    run_dir = new_run_dir("profile", args)
    print(f"saved run metadata to {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(parser)
    parser.add_argument("--tool", choices=["torch", "nsys", "ncu"], default="torch")
    parser.add_argument("--rows", type=int, default=25)
    args = parser.parse_args()

    engine = load_engine(args.engine)
    warmup(engine, args)

    if args.tool == "torch":
        profile_torch(engine, args)
    else:
        profile_external(engine, args)


if __name__ == "__main__":
    main()
