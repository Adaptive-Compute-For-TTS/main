import time
from queue import Queue
from threading import Lock, Thread
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from backend.app import repository
from backend.app.config import PROCESSING_DELAY_SECONDS, WORKER_MODE
from backend.app.database import SessionLocal
from backend.app.models import Run
from backend.app.schemas import QueueStatus, RunCreate


def build_placeholder_result(run: Run) -> Dict[str, Any]:
    """Explicitly opted-in test processor; it does not generate speech."""
    return {
        "message": "Placeholder processing completed",
        "processor": "local-placeholder",
        "received_prompt": run.prompt,
        "nfe": run.nfe,
    }


class QueueUnavailableError(RuntimeError):
    pass


class LocalWorker:
    def __init__(self, mode: str = WORKER_MODE, processor=None) -> None:
        if mode not in {"queue-only", "placeholder", "gpu"}:
            raise ValueError("ADAPTIVE_TTS_WORKER_MODE must be queue-only, gpu or placeholder")
        self.mode = mode
        self._processor = processor
        self._queue: Queue[Optional[str]] = Queue()
        self._thread: Optional[Thread] = None
        self._started = False
        self._lifecycle_lock = Lock()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._started:
                if self.mode == "queue-only" or (self._thread and self._thread.is_alive()):
                    return
            if self._thread is not None and self._thread.is_alive():
                raise QueueUnavailableError("The run queue is still shutting down. Please try again.")

            # Rebuild from SQLite rather than reusing stale items or sentinels.
            pending: Queue[Optional[str]] = Queue()
            with SessionLocal() as database:
                for run_id in repository.recover_pending_runs(database):
                    pending.put(run_id)
            self._queue = pending
            self._thread = None
            self._started = True
            if self.mode != "queue-only":
                self._thread = Thread(target=self._run, name="local-run-worker", daemon=True)
                self._thread.start()

    def _ensure_started(self) -> None:
        if not self._started or (
            self.mode != "queue-only" and (self._thread is None or not self._thread.is_alive())
        ):
            raise QueueUnavailableError("The run queue is unavailable. Please try again.")

    def submit(self, run_id: str) -> None:
        """Enqueue an already persisted UUID, including when no consumer runs."""
        with self._lifecycle_lock:
            self._ensure_started()
            self._queue.put(run_id)

    def create_and_submit(self, database: Session, run_data: RunCreate) -> Run:
        # A shutdown cannot occur between checking availability, committing the
        # durable record, and adding its UUID to the in-process queue.
        with self._lifecycle_lock:
            self._ensure_started()
            run = repository.create_run(database, run_data)
            self._queue.put(run.id)
            return run

    def snapshot(self) -> QueueStatus:
        with self._lifecycle_lock:
            with self._queue.mutex:
                run_ids = [run_id for run_id in self._queue.queue if run_id is not None]
            return QueueStatus(mode=self.mode, count=len(run_ids), run_ids=run_ids)

    def wait_until_idle(self) -> None:
        if self.mode == "queue-only":
            if self.snapshot().count:
                raise RuntimeError("Queue-only mode has no consumer; inspect the queue snapshot instead")
            return
        self._queue.join()

    def stop(self) -> None:
        with self._lifecycle_lock:
            if self._started:
                self._started = False
                if self._thread is not None:
                    self._queue.put(None)
            if self._thread is not None:
                self._thread.join(timeout=5)
                if self._thread.is_alive():
                    # Keep the reference: another start must not create a second
                    # consumer while this one is still finishing its work.
                    raise QueueUnavailableError("The run worker is still shutting down")
            self._thread = None
            self._queue = Queue()

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
            if self.mode == "gpu":
                if self._processor is None:
                    from backend.app.workers.gpu_processor import GpuProcessor
                    self._processor = GpuProcessor()
                outcome = self._processor(run)
                if outcome.status == "completed" and outcome.audio_path:
                    repository.mark_run_completed(database, run_id, outcome.result, outcome.audio_path)
                else:
                    repository.mark_run_failed(database, run_id, outcome.error_message or "Generation returned no validated audio", outcome.result)
                return
            if PROCESSING_DELAY_SECONDS > 0:
                time.sleep(PROCESSING_DELAY_SECONDS)
            result = build_placeholder_result(run)
            repository.mark_run_completed(database, run_id, result)

    def _record_failure(self, run_id: str, error: Exception) -> None:
        with SessionLocal() as database:
            repository.mark_run_failed(database, run_id, str(error))


local_worker = LocalWorker()
