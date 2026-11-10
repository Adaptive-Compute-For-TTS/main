"""Sample GPU utilization, power, clock and temperature on a background thread."""

import statistics
import threading
import time
from pprint import pprint

from .fingerprint import NOT_AVAILABLE
from .toy_model import run_toy_model

DEFAULT_INTERVAL_MS = 50.0
# Metric name -> (summary key prefix, unit suffix). Defines the summary's keys.
METRICS = {
    "util": ("util", "pct"),
    "power": ("power", "w"),
    "sm_clock": ("sm_clock", "mhz"),
    "temp": ("temp", "c"),
}


def summary_keys():
    """The metric keys of summary(), in order (excludes num_samples)."""
    keys = []
    for prefix, unit in METRICS.values():
        keys += [f"{prefix}_mean_{unit}", f"{prefix}_peak_{unit}"]
    return keys


class GpuSampler:
    """Poll NVML every `interval_ms` between start() and stop().

    Works on machines without an NVIDIA GPU: start()/stop() do nothing and
    the summary is filled with placeholders.

    Also usable as a context manager (`with sampler: ...`), which stops the
    thread even if the body raises; read the result with summary() afterwards.
    """

    def __init__(self, interval_ms=DEFAULT_INTERVAL_MS, index=0):
        if interval_ms <= 0:
            raise ValueError(f"interval_ms must be > 0, got {interval_ms}")
        self.interval_ms = float(interval_ms)
        self.interval_s = self.interval_ms / 1000.0
        self.index = index
        # Raw samples from the last run: dicts of t_s plus one value per metric
        # (None where the GPU doesn't report that metric).
        self.samples = []
        self._pynvml = None
        self._handle = None
        self._thread = None
        self._stop_event = threading.Event()

    def start(self):
        if self._thread is not None:
            raise RuntimeError("sampler is already running")
        self.samples = []
        if not self._init_nvml():
            return

        self._stop_event.clear()
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="GpuSampler", daemon=True)
        self._thread.start()

    def stop(self):
        """Stop sampling and return the mean/peak summary."""
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join()
            self._thread = None
            self._pynvml.nvmlShutdown()
            self._pynvml = self._handle = None
        return self.summary()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc_info):
        self.stop()

    def summary(self):
        out = dict.fromkeys(summary_keys(), NOT_AVAILABLE)
        out["num_samples"] = len(self.samples)
        for name, (prefix, unit) in METRICS.items():
            values = [s[name] for s in self.samples if s[name] is not None]
            if values:
                out[f"{prefix}_mean_{unit}"] = statistics.fmean(values)
                out[f"{prefix}_peak_{unit}"] = max(values)
        return out

    def _init_nvml(self):
        """Load NVML and get the device handle; False if there is no usable GPU."""
        try:
            import pynvml
        except ImportError:
            return False

        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError:
            return False

        try:
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
        except pynvml.NVMLError:
            pynvml.nvmlShutdown()
            return False

        self._pynvml = pynvml
        return True

    def _run(self):
        # Sample before the first wait, so even a run shorter than one
        # interval gets at least one sample.
        while True:
            self.samples.append(self._sample())
            # wait() returns True as soon as stop() sets the event, so we
            # never sleep a full interval past stop().
            if self._stop_event.wait(self.interval_s):
                break

    def _sample(self):
        nvml, h = self._pynvml, self._handle
        return {
            "t_s": time.perf_counter() - self._t0,
            "util": self._read(lambda: nvml.nvmlDeviceGetUtilizationRates(h).gpu),
            # NVML reports milliwatts.
            "power": self._read(lambda: nvml.nvmlDeviceGetPowerUsage(h) / 1000.0),
            "sm_clock": self._read(lambda: nvml.nvmlDeviceGetClockInfo(h, nvml.NVML_CLOCK_SM)),
            "temp": self._read(
                lambda: nvml.nvmlDeviceGetTemperature(h, nvml.NVML_TEMPERATURE_GPU)
            ),
        }

    def _read(self, query):
        """Run one NVML query; None if this GPU doesn't support it."""
        try:
            return query()
        except self._pynvml.NVMLError:
            return None


if __name__ == "__main__":
    sampler = GpuSampler()
    sampler.start()
    run_toy_model(32)
    pprint(sampler.stop(), sort_dicts=False)
