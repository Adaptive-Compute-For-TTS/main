import time
from queue import Queue
from threading import Lock, Thread
from typing import Any, Dict, Optional

from backend.app import repository
from backend.app.config import PROCESSING_DELAY_SECONDS
from backend.app.database import SessionLocal
from backend.app.models import Run


def build_placeholder_result(run: Run) -> Dict[str, Any]:
    """Temporary processor contract to be replaced by the GPU pipeline."""
    return {
        "message": "Placeholder processing completed",
        "processor": "local-placeholder",
        "received_prompt": run.prompt,
        "nfe": run.nfe,
    }


class LocalWorker:
    def __init__(self) -> None:
        self._queue: Queue[Optional[str]] = Queue()
        self._thread: Optional[Thread] = None
        self._lifecycle_lock = Lock()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = Thread(
                target=self._run,
                name="local-run-worker",
                daemon=True,
            )
            self._thread.start()

    def submit(self, run_id: str) -> None:
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("Local worker is not running")
        self._queue.put(run_id)

    def wait_until_idle(self) -> None:
        self._queue.join()

    def stop(self) -> None:
        with self._lifecycle_lock:
            if self._thread is None:
                return
            self._queue.put(None)
            self._thread.join(timeout=5)
            self._thread = None

    def _run(self) -> None:
        while True:
            run_id = self._queue.get()
            try:
                if run_id is None:
                    return
                self._process(run_id)
            except Exception as error:
                if run_id is not None:
                    self._record_failure(run_id, error)
            finally:
                self._queue.task_done()

    def _process(self, run_id: str) -> None:
        with SessionLocal() as database:
            run = repository.mark_run_running(database, run_id)
            if run is None:
                return

            if PROCESSING_DELAY_SECONDS > 0:
                time.sleep(PROCESSING_DELAY_SECONDS)

            result = build_placeholder_result(run)
            repository.mark_run_completed(database, run_id, result)

    def _record_failure(self, run_id: str, error: Exception) -> None:
        with SessionLocal() as database:
            repository.mark_run_failed(database, run_id, str(error))


local_worker = LocalWorker()
