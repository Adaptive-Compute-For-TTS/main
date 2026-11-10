"""Run one job through the real worker by hand: load dots.tts, warm up, generate, validate, store.

The check that the worker works on this GPU before the queue is wired in. Prints the JobOutcome
(what would go into the runs row) as JSON and exits 1 if the run failed.

Settings come from GPU_WORKER_* environment variables (see src/gpu_worker/config.py), e.g.:
    python scripts/run_one.py --prompt "Hello there." --nfe 10
    GPU_WORKER_MODEL=dots-studio/dots.tts-base python scripts/run_one.py --prompt "Hello there." --nfe 4
    GPU_WORKER_OPTIMIZE=1 python scripts/run_one.py --prompt "Hello there." --nfe 10   # ~3 min compile
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import uuid

from gpu_worker import GpuWorker, Job, WorkerConfig


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--nfe", type=int, required=True)
    p.add_argument("--run-id", default=None, help="default: a new UUID, like the backend's run ids")
    p.add_argument("--no-warmup", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = WorkerConfig.from_env()
    if a.no_warmup:
        config = dataclasses.replace(config, warmup_nfes=())
    worker = GpuWorker.load(config)
    outcome = worker.run_job(Job(a.run_id or str(uuid.uuid4()), a.prompt, a.nfe))
    print(json.dumps(dataclasses.asdict(outcome), indent=2, default=str))
    return 0 if outcome.ok else 1


if __name__ == "__main__":
    sys.exit(main())
