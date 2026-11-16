"""Adapters to the team's telemetry harness and per-run quality evaluator.

These modules live in sibling monorepo folders, rather than installed packages.
Imports and model loading stay lazy, so queue-only mode never loads torch/ASR.
"""
import importlib
import importlib.util
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[3]


def _telemetry_module(name):
    package = "_adaptive_gpu_telemetry"
    if package not in sys.modules:
        folder = ROOT / "gpu-telemetry" / "telemetry"
        spec = importlib.util.spec_from_file_location(
            package, folder / "__init__.py", submodule_search_locations=[str(folder)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(package, None)
            raise
    return importlib.import_module(f"{package}.{name}")


class GenerationTelemetry:
    """Measure exactly one generation; instrumentation cannot trigger a retry."""

    def measure(self, generate, nfe):
        import torch

        report = {
            "status": "collected", "scope": "generation_only", "errors": [],
            "fingerprint": {"device": "cuda" if torch.cuda.is_available() else "cpu",
                            "python_version": platform.python_version(), "torch_version": str(torch.__version__)},
        }
        state = {}

        def once():
            if state.get("called"):
                raise RuntimeError("Telemetry attempted to generate more than once")
            state["called"] = True
            started = time.perf_counter()
            try:
                state["value"] = generate()
            except Exception as error:
                state["error"] = error
                raise
            finally:
                state["elapsed"] = time.perf_counter() - started

        memory = sampler = None
        try:
            report["fingerprint"] = _telemetry_module("fingerprint").get_fingerprint()
            memory = _telemetry_module("memory")
            memory.reset()
            sampler = _telemetry_module("nvml_sampler").GpuSampler()
            report["timing"] = _telemetry_module("timing").time_callable(
                once, nfe, warmups=0, repeats=1, measure=sampler
            )
        except Exception as error:
            if "error" in state:
                raise state["error"]
            report["errors"].append(f"Telemetry: {type(error).__name__}: {error}")
            if not state.get("called"):
                once()
        finally:
            if sampler is not None:
                try:
                    sampler.stop()
                    report["gpu"] = {**sampler.summary(), "interval_ms": sampler.interval_ms}
                except Exception as error:
                    report["errors"].append(f"GPU sampling: {type(error).__name__}: {error}")
        if memory is not None:
            try:
                report["memory"] = memory.read_peak()
            except Exception as error:
                report["errors"].append(f"Memory: {type(error).__name__}: {error}")
        report.setdefault("timing", {"device": report["fingerprint"]["device"], "timer": "perf_counter"})
        report["timing"]["wall_time_s"] = state["elapsed"]
        if report["fingerprint"]["device"] == "cpu":
            report["gpu_note"] = "NVIDIA GPU measurements are unavailable on this CPU run."
        elif not report.get("gpu", {}).get("num_samples"):
            report["gpu_note"] = "No NVML samples were available; GPU sampling requires a working NVIDIA driver and NVML."
        if report["errors"]:
            report["status"] = "partial"
        return state["value"], report


class RunQuality:
    """Reuse the team's evaluator and cached ASR model across queued runs."""

    def __init__(self):
        self.reference = os.getenv("ADAPTIVE_TTS_QUALITY_REFERENCE_WAV") or None
        self.model = os.getenv("ADAPTIVE_TTS_ASR_MODEL", "small")
        self.evaluator = None

    def _load(self):
        if self.evaluator is not None:
            return
        folder = ROOT / "quality-evals" / "speaker_sim"
        # The team's evaluator currently uses sibling script imports.
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
        module_name = "_adaptive_run_quality"
        if module_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(module_name, folder / "evaluate_run.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[module_name] = module
        module = sys.modules[module_name]
        module.asr.set_model(self.model)
        self.evaluator = module.QualityEvaluator(reference_wav=self.reference)

    def __call__(self, audio_path, prompt, validation):
        if not validation.get("ok"):
            return {"status": "skipped", "scored": False, "errors": [],
                    "notes": ["Quality evaluation skipped because WAV validation failed."]}
        started = time.perf_counter()
        try:
            self._load()
            result = self.evaluator.evaluate(audio_path, prompt, validation=validation)
            result["status"] = "partial" if result.get("scored") and result.get("errors") else "completed" if result.get("scored") else "unavailable"
        except Exception as error:
            result = {"status": "unavailable", "scored": False, "wer": None, "cer": None,
                      "sim_ref": None, "errors": [f"Quality evaluation: {type(error).__name__}: {error}"],
                      "notes": ["Check the worker quality dependencies and model download access, then submit a new run."]}
        result["evaluation_time_s"] = round(time.perf_counter() - started, 4)
        result["wer_version"] = "no_intro, capped at 1.0"
        result["asr_settings"] = {"model": self.model, "device": "cpu", "compute_type": "int8", "language": "en"}
        if not self.reference:
            result.setdefault("notes", []).append("Speaker similarity skipped: no reference recording is configured.")
        return result
