"""Lazy boundary to the monorepo's real worker, executor and WAV validator.

No torch/model import occurs in queue-only mode or ordinary API tests.
"""
from dataclasses import replace
from backend.app.config import AUDIO_DIR


class GpuProcessor:
    def __init__(self):
        from gpu_worker import GpuWorker, WorkerConfig
        from backend.app.workers.diagnostics import GenerationTelemetry, RunQuality
        config = replace(WorkerConfig.from_env(), storage_dir=AUDIO_DIR)
        self.worker = GpuWorker.load(config)
        self.worker.telemetry = GenerationTelemetry()
        self.worker.quality = RunQuality()

    def __call__(self, run):
        from gpu_worker import Job
        return self.worker.run_job(Job(run_id=run.id, prompt=run.prompt, nfe=run.nfe))
