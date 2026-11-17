"""Minimal name -> factory registry used for models, backends and probes.

Entries are registered lazily as "module:attr" strings so that importing the registry never
pulls in heavy optional dependencies (torch, modal, dots_tts).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, str | Callable[..., T]] = {}

    def register(self, name: str, target: str | Callable[..., T]) -> None:
        if name in self._entries:
            raise ValueError(f"{self._kind} {name!r} is already registered")
        self._entries[name] = target

    def names(self) -> list[str]:
        return sorted(self._entries)

    def resolve(self, name: str) -> Callable[..., T]:
        try:
            target = self._entries[name]
        except KeyError:
            raise KeyError(f"unknown {self._kind} {name!r}; available: {', '.join(self.names())}") from None
        if isinstance(target, str):
            module_name, _, attr = target.partition(":")
            target = getattr(importlib.import_module(module_name), attr)
            self._entries[name] = target
        return target

    def create(self, name: str, **params: object) -> T:
        return self.resolve(name)(**params)
