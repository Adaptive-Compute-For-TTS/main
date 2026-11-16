#!/usr/bin/env python3
"""Run the local UI and FastAPI server; choose queue-only inspection or real generation."""

import argparse
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent.parent


def port_number(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def require_free_port(port: int) -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            # Match the servers' reuse policy: a recently closed connection in
            # TIME_WAIT is not a running service and must not block restart.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))
    except OSError as error:
        raise RuntimeError(
            f"Port {port} is already in use. Stop that service or choose another port."
        ) from error


def stop_process(process: subprocess.Popen) -> None:
    # Each child has its own session; also stop npm's Next.js child.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def wait_for_backend(process: subprocess.Popen, url: str) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("FastAPI exited during startup; see its output above.")
        try:
            with urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            pass
        time.sleep(0.2)
    raise RuntimeError("FastAPI did not become healthy within 20 seconds.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-mode", choices=["queue-only", "gpu"], default="queue-only")
    parser.add_argument("--python", type=Path, help="Python environment with backend and GPU packages installed")
    parser.add_argument("--backend-port", type=port_number, default=8001)
    parser.add_argument("--frontend-port", type=port_number, default=3005)
    args = parser.parse_args()
    python = args.python.resolve() if args.python else ROOT / ".venv" / "bin" / "python"
    frontend = ROOT / "frontend"
    npm = shutil.which("npm")
    children = []

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.backend_port == args.frontend_port:
            raise RuntimeError("The frontend and backend need different ports.")
        if not python.is_file():
            raise RuntimeError(
                "Create the project environment first: python3 -m venv .venv, then "
                ".venv/bin/python -m pip install -r backend/requirements.txt"
            )
        if not npm:
            raise RuntimeError("Install Node.js with npm before starting the frontend.")
        if not (frontend / "node_modules" / "next").is_dir():
            raise RuntimeError("Install frontend dependencies first: npm --prefix frontend ci")
        require_free_port(args.backend_port)
        require_free_port(args.frontend_port)

        backend_url = f"http://127.0.0.1:{args.backend_port}"
        frontend_url = f"http://127.0.0.1:{args.frontend_port}"
        backend_env = os.environ.copy()
        backend_env["ADAPTIVE_TTS_WORKER_MODE"] = args.worker_mode
        backend = subprocess.Popen(
            [str(python), "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1",
             "--port", str(args.backend_port)],
            cwd=ROOT, env=backend_env, start_new_session=True,
        )
        children.append(backend)
        wait_for_backend(backend, backend_url + "/health")

        frontend_env = os.environ.copy()
        frontend_env["FASTAPI_BASE_URL"] = backend_url
        ui = subprocess.Popen(
            [npm, "run", "dev", "--", "--hostname", "127.0.0.1", "--port", str(args.frontend_port)],
            cwd=frontend, env=frontend_env, start_new_session=True,
        )
        children.append(ui)
        print(
            f"\nAdaptive Compute\n"
            f"  Frontend: {frontend_url}/playground\n"
            f"  FastAPI:  {backend_url}/docs\n"
            f"  Queue:    {backend_url}/queue\n"
            f"  Worker mode: {args.worker_mode}.\n"
            "  No runs are created at startup. Press Ctrl+C to stop both services.\n",
            flush=True,
        )
        while True:
            for name, process in (("FastAPI", backend), ("Frontend", ui)):
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited (code {process.returncode}); stopping both services.")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopping local services…", flush=True)
        return 0
    except (OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        for process in reversed(children):
            stop_process(process)


if __name__ == "__main__":
    raise SystemExit(main())
