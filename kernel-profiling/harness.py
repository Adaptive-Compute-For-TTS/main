"""Shared bits for bench.py and profiler.py: CLI args, workloads, run archiving."""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from tts import ENGINES, TTSEngine

RUNS_DIR = Path(__file__).parent / "runs"

TEXTS = {
    "short": "Hi there",
    "sentence": "The quick brown fox jumps over the lazy dog, then naps in the afternoon sun.",
    "paragraph": (
        "Profiling a text to speech model is mostly about finding where the time "
        "actually goes. The language model, the flow matching decoder, and the "
        "vocoder each have very different bottlenecks, so a single number rarely "
        "tells the whole story. Measure first, then optimize the biggest slice."
    ),
}


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", choices=sorted(ENGINES), default="torch-compiled")
    parser.add_argument("--mode", choices=["generate", "stream"], default="stream")
    parser.add_argument("--text", choices=sorted(TEXTS), default="short")
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--tag", default="", help="free-form note saved with the run")
    parser.add_argument("--seed", type=int, default=0, help="reseeded before every utterance")


def gen_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return dict(text=TEXTS[args.text], num_steps=args.num_steps)


def reseed(args: argparse.Namespace) -> None:
    # same noise every utterance -> same audio length, and outputs of two
    # engines can be diffed sample by sample
    torch.manual_seed(args.seed)


def run_once(engine: TTSEngine, mode: str, *, profiled: bool, **kwargs: Any) -> torch.Tensor:
    """Run one utterance to completion, return the audio."""
    if mode == "generate":
        fn = engine.profile if profiled else engine.generate
        audio = fn(**kwargs)
    else:
        fn = engine.profile_stream if profiled else engine.generate_stream
        audio = torch.cat(list(fn(**kwargs)), dim=-1)
    torch.cuda.synchronize()
    return audio


def warmup(engine: TTSEngine, args: argparse.Namespace) -> None:
    for i in range(args.warmup):
        print(f"warmup {i + 1}/{args.warmup}")
        reseed(args)
        run_once(engine, args.mode, profiled=False, **gen_kwargs(args))


def _git(*cmd: str) -> str:
    try:
        return subprocess.check_output(["git", *cmd], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def new_run_dir(kind: str, args: argparse.Namespace, **extra: Any) -> Path:
    """Create runs/<kind>/<timestamp>_<engine>_<mode>_<text>/ with a meta.json."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / kind / f"{stamp}_{args.engine}_{args.mode}_{args.text}"
    run_dir.mkdir(parents=True)
    meta = {
        "kind": kind,
        "args": vars(args),
        "kwargs": gen_kwargs(args),
        "git_commit": _git("rev-parse", "HEAD"),
        # untracked files (like runs/ itself) don't count as dirty
        "git_dirty": _git("status", "--porcelain", "--untracked-files=no") != "",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        **extra,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return run_dir
