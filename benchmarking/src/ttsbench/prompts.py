"""Benchmark prompt sets: versioned JSONL files under benchmarks/prompts/.

A prompt set is append-only once published; changing text means a new file (v1, v2, ...),
so a run's `prompts_hash` identifies exactly which inputs produced its numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

from ttsbench.config import stable_hash
from ttsbench.schema import Prompt, PromptSelection


def load_prompt_file(path: Path) -> list[Prompt]:
    prompts = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            prompts.append(Prompt.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as err:
            raise ValueError(f"{path}:{lineno}: invalid prompt: {err}") from err
    ids = [p.id for p in prompts]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"{path}: duplicate prompt ids: {duplicates}")
    return prompts


def select_prompts(selection: PromptSelection, root: Path = Path(".")) -> list[Prompt]:
    prompts = load_prompt_file(root / selection.path)
    if selection.ids:
        missing = set(selection.ids) - {p.id for p in prompts}
        if missing:
            raise ValueError(f"unknown prompt ids: {sorted(missing)}")
        prompts = [p for p in prompts if p.id in selection.ids]
    if selection.lengths:
        prompts = [p for p in prompts if p.length in selection.lengths]
    if selection.tags:
        prompts = [p for p in prompts if set(selection.tags) & set(p.tags)]
    if selection.limit is not None:
        prompts = prompts[: selection.limit]
    if not prompts:
        raise ValueError("prompt selection matched no prompts")
    return prompts


def prompts_hash(prompts: list[Prompt]) -> str:
    return stable_hash([p.model_dump(mode="json") for p in prompts])
