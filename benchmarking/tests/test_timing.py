from __future__ import annotations

import gc
import time

import pytest

from ttsbench.measure.timing import Device, StageTimer, host_timed, measure


def test_measure_records_wall_time_and_stages() -> None:
    def work(timer: StageTimer) -> str:
        for _ in range(3):
            with timer.stage("step"):
                time.sleep(0.002)
        return "done"

    result, timing = measure(work, Device("cpu"))

    assert result == "done"
    assert timing.device_ms is None
    assert timing.stage_calls == {"step": 3}
    assert timing.stages_ms["step"] >= 6.0
    assert timing.latency_ms >= timing.stages_ms["step"]


def test_measure_restores_gc_even_on_error() -> None:
    assert gc.isenabled()

    def boom(timer: StageTimer) -> None:
        assert not gc.isenabled()
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError):
        measure(boom, Device("cpu"))
    assert gc.isenabled()


def test_device_synchronize_brackets_the_timed_region() -> None:
    calls: list[str] = []

    class RecordingDevice(Device):
        def synchronize(self) -> None:
            calls.append("sync")

    def work(timer: StageTimer) -> None:
        calls.append("work")

    measure(work, RecordingDevice("cpu"))
    assert calls == ["sync", "work", "sync"]


def test_unknown_device_rejected() -> None:
    with pytest.raises(ValueError):
        Device("tpu")


def test_host_timed() -> None:
    value, ms = host_timed(lambda: time.sleep(0.001) or 7)
    assert value == 7 and ms >= 1.0
