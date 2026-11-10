"""GPU worker: turns a queued run (run_id, prompt, nfe) into a validated WAV plus the run's result.

    from gpu_worker import GpuWorker, Job, WorkerConfig
    worker = GpuWorker.load(WorkerConfig.from_env())     # loads dots.tts once, warms up
    outcome = worker.run_job(Job(run_id, prompt, nfe))   # never raises for a failed run
    outcome.status, outcome.result, outcome.audio_path, outcome.error_message   # -> the runs row

The queue loop (claiming runs, updating SQLite) lives outside this package.
"""

__version__ = "0.1.0"

from gpu_worker.config import WorkerConfig  # noqa: E402
from gpu_worker.worker import GpuWorker, Job, JobOutcome  # noqa: E402

__all__ = ["GpuWorker", "Job", "JobOutcome", "WorkerConfig", "__version__"]
