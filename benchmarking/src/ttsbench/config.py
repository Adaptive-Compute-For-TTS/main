"""Experiment config loading, CLI overrides and stable hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from ttsbench.schema import ExperimentConfig


def load_config(path: Path, overrides: list[str] | None = None) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    for override in overrides or []:
        raw = apply_override(raw, override)
    return ExperimentConfig.model_validate(raw)


def apply_override(raw: dict[str, Any], override: str) -> dict[str, Any]:
    """`a.b.c=value` where value is parsed as YAML (so `sweep.nfe=[4,8]` works)."""
    dotted, sep, value = override.partition("=")
    if not sep or not dotted:
        raise ValueError(f"override must look like key.path=value, got {override!r}")
    keys = dotted.split(".")
    return _set_path(raw, keys, yaml.safe_load(value))


def _set_path(node: dict[str, Any], keys: list[str], value: Any) -> dict[str, Any]:
    head, rest = keys[0], keys[1:]
    if not rest:
        return {**node, head: value}
    child = node.get(head) or {}
    if not isinstance(child, dict):
        raise ValueError(f"cannot set {'.'.join(keys)}: {head!r} is not a mapping")
    return {**node, head: _set_path(child, rest, value)}


def stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode()).hexdigest()


def config_hash(config: ExperimentConfig) -> str:
    """Hash of everything that affects results (output location excluded)."""
    return stable_hash(config.model_dump(mode="json", exclude={"output_dir", "description"}))
