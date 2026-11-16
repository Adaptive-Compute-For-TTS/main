#!/usr/bin/env python3
"""Display the actual in-memory queue and recent SQLite runs using read-only API calls."""

import argparse
from datetime import datetime
import json
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def positive_interval(value: str) -> float:
    interval = float(value)
    if interval < 0.1:
        raise argparse.ArgumentTypeError("interval must be at least 0.1 seconds")
    return interval


def row_limit(value: str) -> int:
    limit = int(value)
    if not 1 <= limit <= 100:
        raise argparse.ArgumentTypeError("limit must be between 1 and 100")
    return limit


def read_json(url: str):
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=3) as response:
        return json.load(response)


def printable(value) -> str:
    return " ".join("".join(character if character.isprintable() else " " for character in str(value)).split())


def display(base: str, limit: int, clear: bool) -> None:
    queue = read_json(base + "/queue")
    runs = read_json(base + f"/runs?limit={limit}&offset=0")
    if clear:
        print("\033[2J\033[H", end="")
    print(f"Adaptive Compute queue — {datetime.now().astimezone().strftime('%H:%M:%S %Z')}")
    print(f"Mode: {printable(queue['mode'])} | Waiting in queue.Queue: {queue['count']} | Recent SQLite runs: {len(runs)}")
    print()
    if not runs:
        print("No runs. Submit text using the frontend arrow button.")
        return
    waiting = set(queue["run_ids"])
    print(f"{'RUN ID':36}  {'NFE':>3}  {'STATUS':10}  {'IN QUEUE':8}  PROMPT")
    print("─" * 115)
    for run in runs:
        prompt = printable(run["prompt"])
        if len(prompt) > 68:
            prompt = prompt[:67] + "…"
        print(f"{printable(run['id']):36}  {run['nfe']:>3}  {printable(run['status']):10}  "
              f"{'yes' if run['id'] in waiting else 'no':8}  {prompt}")
    print("\nRead-only snapshot; this command does not dequeue or submit jobs.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-url", default="http://127.0.0.1:8001")
    parser.add_argument("--limit", type=row_limit, default=20)
    parser.add_argument("--watch", action="store_true", help="refresh until Ctrl+C")
    parser.add_argument("--interval", type=positive_interval, default=1.0)
    args = parser.parse_args()
    base = args.backend_url.rstrip("/")
    parsed = urlsplit(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
        parser.error("backend URL must be an HTTP(S) base URL without a query or fragment")
    try:
        while True:
            try:
                display(base, args.limit, args.watch and sys.stdout.isatty())
            except (HTTPError, URLError, OSError, ValueError, KeyError, TypeError) as error:
                if isinstance(error, HTTPError):
                    message = f"backend returned HTTP {error.code}"
                else:
                    message = "could not read the backend; check that scripts/dev.py is running"
                print(f"Error: {message}.", file=sys.stderr, flush=True)
                if not args.watch:
                    return 1
            if not args.watch:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
