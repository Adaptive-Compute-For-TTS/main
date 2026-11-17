"""ttsbench command line: run, ls, summarize, compare, push, eval."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ttsbench.compare import compare_runs
from ttsbench.config import load_config
from ttsbench.publish import check_publish_config, publish_run
from ttsbench.runner import run_experiment
from ttsbench.storage import Run, list_runs
from ttsbench.summary import flatten_row


def _table(rows: list[dict[str, Any]], columns: Sequence[str]) -> str:
    def fmt(value: Any) -> str:
        return f"{value:.4g}" if isinstance(value, float) else str(value)

    cells = [[fmt(r.get(c, "")) for c in columns] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(columns)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(columns, widths, strict=True))]
    lines += ["  ".join(v.ljust(w) for v, w in zip(row, widths, strict=True)) for row in cells]
    return "\n".join(lines)


def _print_summary(run: Run) -> None:
    rows = [flatten_row(r) for r in run.summary()]
    cols = (
        "nfe",
        "length",
        "n",
        "latency_p50_ms",
        "latency_p90_ms",
        "within_prompt_cv",
        "rtf_p50",
        "deterministic_fraction",
        "noisy",
    )
    print(_table(rows, cols))
    if any(r["noisy"] for r in rows):
        print("\nwarning: per-prompt latency CV > 5% in some groups; check for GPU contention or raise warmup.")


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.set)
    if args.push:
        check_publish_config()
    run = run_experiment(config, root=Path.cwd())
    print(f"\nrun stored at {run.path}\n")
    _print_summary(run)
    if args.push:
        print(f"\npublished to {publish_run(run)}")
    return 0


def cmd_ls(args: argparse.Namespace) -> int:
    rows = []
    for run in list_runs(args.runs_dir):
        m = run.manifest
        rows.append(
            {
                "run": run.path.name,
                "status": m["status"],
                "model": m["config"]["model"]["name"],
                "backend": m["backend"].get("name"),
                "gpu": m["backend"].get("gpu", "-"),
                "nfe": ",".join(map(str, m["config"]["sweep"]["nfe"])),
                "git": (m["git"].get("sha") or "-")[:8] + ("*" if m["git"].get("dirty") else ""),
            }
        )
    print(_table(rows, ("run", "status", "model", "backend", "gpu", "nfe", "git")) if rows else "no runs")
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    run = Run(args.run)
    run.write_summary()
    _print_summary(run)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    result = compare_runs(Run(args.a), Run(args.b))
    for warning in result.warnings:
        print(f"warning: {warning}")
    cols = ("nfe", "a_p50_ms", "b_p50_ms", "speedup_p50", "a_p90_ms", "b_p90_ms", "audio_identical")
    print(_table(result.rows, cols))
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    for path in args.runs:
        print(f"{path.name}: {publish_run(Run(path), allow_incomplete=args.allow_incomplete)}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from ttsbench.evals import run_eval_on_modal

    if not args.eval_command:
        raise SystemExit("ttsbench eval: give the command after `--`")
    record = run_eval_on_modal(
        args.bundle, args.eval_command, args.out, requirements=args.requirements, gpu=args.gpu, data=args.data
    )
    print(record["stdout"][-4000:])
    if record["returncode"] != 0:
        print(record["stderr"][-4000:], file=sys.stderr)
    print(f"eval exited {record['returncode']} in {record['wall_time_s']}s; outputs in {args.out}")
    return record["returncode"]


def cmd_manifest(args: argparse.Namespace) -> int:
    from ttsbench.evals import write_eval_manifest

    print(write_eval_manifest(Run(args.run), args.out))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ttsbench", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run an experiment config")
    p.add_argument("config", type=Path)
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set sweep.nfe=[4,8] --set backend.params.gpu=A100",
    )
    p.add_argument("--push", action="store_true", help="publish to S3 + Postgres when the run completes")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("push", help="publish runs to object storage + Postgres (see ttsbench.publish)")
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--allow-incomplete", action="store_true", help="also publish failed/running runs")
    p.set_defaults(func=cmd_push)

    p = sub.add_parser("eval", help="run an eval bundle on Modal: ttsbench eval BUNDLE --out DIR -- CMD ... {out} ...")
    p.add_argument("bundle", type=Path)
    p.add_argument("--out", type=Path, required=True, help="local folder for the eval's outputs")
    p.add_argument("--requirements", default="requirements.txt", help="path inside the bundle")
    p.add_argument("--gpu", default=None, help="Modal GPU type; CPU if omitted")
    p.add_argument("--data", type=Path, default=None, help="extra folder mounted at /data (e.g. a run directory)")
    p.set_defaults(func=cmd_eval, eval_command=[])

    p = sub.add_parser("manifest", help="write <run>/eval_manifest.csv (clip_id, audio_path, reference_text, ...)")
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("ls", help="list stored runs")
    p.add_argument("runs_dir", type=Path, nargs="?", default=Path("runs"))
    p.set_defaults(func=cmd_ls)

    p = sub.add_parser("summarize", help="(re)compute and print a run's summary")
    p.add_argument("run", type=Path)
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("compare", help="compare run B against baseline run A")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.set_defaults(func=cmd_compare)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Everything after `--` is the eval command, passed through untouched (it has its own flags).
    tail = argv[argv.index("--") + 1 :] if "--" in argv else []
    head = argv[: argv.index("--")] if "--" in argv else argv
    args = build_parser().parse_args(head)
    if tail:
        args.eval_command = tail
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
