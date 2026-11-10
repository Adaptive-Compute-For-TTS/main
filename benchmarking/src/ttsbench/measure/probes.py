"""Hook for per-sample telemetry, run inside the worker around (not inside) the timed region.

This is the seam with GPU telemetry: implement `Probe`, register it in `PROBES`, and list its
name under `probes:` in an experiment config. Whatever `after()` returns is stored in
`Sample.telemetry[<probe name>]`. No probes ship with the benchmark harness itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ttsbench.measure.timing import Device
from ttsbench.registry import Registry


class Probe(ABC):
    def __init__(self, device: Device) -> None:
        self.device = device

    def before(self) -> None:  # noqa: B027 - optional hook
        """Called just before the device sync that opens the timed region."""

    @abstractmethod
    def after(self) -> dict[str, Any]:
        """Called after the timed region has closed (device already synchronised)."""


PROBES: Registry[Probe] = Registry("probe")
