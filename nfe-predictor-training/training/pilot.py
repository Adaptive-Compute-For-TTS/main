"""Pure, CPU-testable contracts for controlled single-patch NFE comparisons.

One *comparison* contains one prompt, seed, speaker and target patch, with
several NFE candidates run from an identical pre-patch state and noise.
All quality labels remain unknown until an explicit scoring protocol is run.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = 'patch-nfe-v1'


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def select_patch_indices(num_patches: int, how_many: int = 3) -> tuple[int, ...]:
    """Early/mid/late positions; never repeat a target on short clips."""
    if num_patches < 1 or how_many < 1:
        raise ValueError('num_patches and how_many must be positive')
    n = min(num_patches, how_many)
    if n == 1:
        return (0,)
    return tuple(sorted({round(i * (num_patches - 1) / (n - 1)) for i in range(n)}))


@dataclass(frozen=True)
class PilotPlan:
    prompt_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    budgets: tuple[int, ...]
    reference_nfe: int
    speaker_id: str
    target_patches: int = 3

    def __post_init__(self):
        if not self.prompt_ids or len(set(self.prompt_ids)) != len(self.prompt_ids):
            raise ValueError('prompt_ids must be unique and nonempty')
        if not self.seeds or any(type(x) is not int for x in self.seeds) or len(set(self.seeds)) != len(self.seeds):
            raise ValueError('seeds must be distinct integers')
        if not self.budgets or tuple(sorted(set(self.budgets))) != self.budgets or any(type(x) is not int or x < 1 for x in self.budgets):
            raise ValueError('budgets must be unique, positive, ascending ints')
        if type(self.reference_nfe) is not int or self.reference_nfe not in self.budgets:
            raise ValueError('reference_nfe must be a candidate budget')
        if (not isinstance(self.speaker_id, str) or not self.speaker_id.strip()
                or type(self.target_patches) is not int or self.target_patches < 1):
            raise ValueError('speaker_id and target_patches are required')

    @classmethod
    def from_json(cls, path: str | Path) -> 'PilotPlan':
        obj = json.loads(Path(path).read_text(encoding='utf-8'))
        if obj.get('schema_version') != SCHEMA:
            raise ValueError('Unexpected pilot schema')
        return cls(prompt_ids=tuple(obj['prompt_ids']), seeds=tuple(obj['seeds']),
                   budgets=tuple(obj['budgets']), reference_nfe=obj['reference_nfe'],
                   speaker_id=obj['speaker_id'], target_patches=obj.get('target_patches', 3))


def comparison_id(prompt_id: str, seed: int, speaker_id: str, patch_index: int,
                  conditioning_id: str, experiment_id: str | None = None) -> str:
    return 'patch-' + stable_hash([prompt_id, seed, speaker_id, patch_index, conditioning_id, experiment_id])[:20]


def matching_prefix(reference: Mapping[str, Any], candidate: Mapping[str, Any], target: int) -> list[str]:
    """Return reasons a comparison is NOT a controlled single-patch change.

    Every prior patch's input, sampled noise and output must match. The target
    patch must start with identical input and noise, but its output may differ.
    """
    errors: list[str] = []
    if type(target) is not int or target < 0:
        raise ValueError('target must be a nonnegative patch index')
    r_history, c_history = reference['nfe_per_patch'], candidate['nfe_per_patch']
    r_patches, c_patches = reference['patches'], candidate['patches']
    if len(r_history) <= target or len(c_history) <= target or len(r_patches) <= target or len(c_patches) <= target:
        return ['target_patch_not_generated']
    if len(r_history) != len(r_patches) or len(c_history) != len(c_patches):
        errors.append('incomplete_patch_recording')
    if r_history[:target] != c_history[:target]:
        errors.append('preceding_nfe_schedule_differs')
    if any(n != reference['reference_nfe'] for n in r_history):
        errors.append('reference_schedule_not_fixed')
    if any(n != reference['reference_nfe'] for i, n in enumerate(c_history) if i != target):
        errors.append('non_target_nfe_differs')
    for i in range(target):
        for k in ('anchor_state_hash', 'noise_hash', 'patch_output_hash'):
            if not r_patches[i].get(k) or not c_patches[i].get(k):
                errors.append(f'preceding_patch_{i}_{k}_missing')
            elif r_patches[i][k] != c_patches[i][k]:
                errors.append(f'preceding_patch_{i}_{k}_differs')
    for k in ('anchor_state_hash', 'noise_hash'):
        if not r_patches[target].get(k) or not c_patches[target].get(k):
            errors.append(f'target_{k}_missing')
        elif r_patches[target][k] != c_patches[target][k]:
            errors.append(f'target_{k}_differs')
    for i in range(target + 1):
        if r_patches[i].get('llm_hidden_hash') != c_patches[i].get('llm_hidden_hash'):
            errors.append(f'patch_{i}_llm_hidden_hash_differs')
    return errors


def build_observation(
    *, prompt: Mapping[str, Any], seed: int, speaker_id: str, patch_index: int,
    budget: int, candidate: Mapping[str, Any], reference: Mapping[str, Any],
    checkpoint_id: str, conditioning_id: str, experiment: Mapping[str, Any] | None = None,
    recorder_equivalence: bool = False,
) -> dict[str, Any]:
    errors = matching_prefix(reference, candidate, patch_index)
    if len(candidate['nfe_per_patch']) > patch_index and candidate['nfe_per_patch'][patch_index] != budget:
        errors.append('candidate_target_nfe_differs')
    rec = candidate['patches'][patch_index] if len(candidate['patches']) > patch_index else {}
    experiment_id = stable_hash(experiment) if experiment else None
    return {
        'schema_version': SCHEMA,
        'comparison_id': comparison_id(prompt['id'], seed, speaker_id, patch_index, conditioning_id, experiment_id),
        'experiment_id': experiment_id,
        'generation_config': dict(experiment) if experiment else None,
        'solver': 'euler',
        'recorder_equivalence': recorder_equivalence,
        'prompt_id': prompt['id'],
        'prompt_family_id': prompt.get('prompt_family_id', prompt['id']),
        'speaker_id': speaker_id,
        'text': prompt['text'],
        'patch_index': patch_index,
        'nfe': budget,
        'seed': seed,
        'checkpoint_id': checkpoint_id,
        'conditioning_id': conditioning_id,
        'reference_nfe': reference['reference_nfe'],
        'preceding_nfe_schedule': list(reference['nfe_per_patch'][:patch_index]),
        'anchor_state_hash': rec.get('anchor_state_hash'),
        'noise_hash': rec.get('noise_hash'),
        'prepatch_llm_hidden_hash': rec.get('llm_hidden_hash'),
        'patch_output_hash': rec.get('patch_output_hash'),
        'candidate_audio_path': candidate.get('audio_path'),
        'reference_audio_path': reference.get('audio_path'),
        'candidate_audio_sha256': candidate.get('audio_sha256'),
        'reference_audio_sha256': reference.get('audio_sha256'),
        'candidate_sample_rate': candidate.get('sample_rate'),
        'reference_sample_rate': reference.get('sample_rate'),
        'candidate_num_patches': len(candidate['nfe_per_patch']),
        'reference_num_patches': len(reference['nfe_per_patch']),
        'candidate_max_generate_length': candidate.get('max_generate_length'),
        'reference_max_generate_length': reference.get('max_generate_length'),
        'candidate_duration_s': candidate.get('audio_s'),
        'reference_duration_s': reference.get('audio_s'),
        'candidate_generation_s': candidate.get('time_s'),
        'reference_generation_s': reference.get('time_s'),
        'matched_context': not errors,
        'match_errors': errors,
        # Filled only by the scoring step, never by the generation step.
        'candidate_validation': None,
        'reference_validation': None,
        'candidate_quality': None,
        'reference_quality': None,
        'quality_pass': None,
        'label_protocol': None,
    }


def validate_observations(observations: Sequence[Mapping[str, Any]], budgets: tuple[int, ...]) -> None:
    """Ensure valid comparisons before loading supervised targets."""
    from training.contracts import validate_budgets, validate_observation

    validate_budgets(budgets)
    seen: set[tuple[str, int]] = set()
    by_comparison: dict[str, list[Mapping[str, Any]]] = {}
    for row in observations:
        if row.get('quality_pass') is not None and row.get('matched_context') is not True:
            raise ValueError('Unmatched observation cannot have a quality label')
        validate_observation(dict(row), budgets)
        if type(row.get('matched_context')) is not bool:
            raise ValueError('matched_context must be a boolean')
        if row.get('schema_version') != SCHEMA:
            raise ValueError('Schema mismatch')
        key = (row['comparison_id'], row['nfe'])
        if key in seen:
            raise ValueError(f'duplicate comparison/budget: {key}')
        seen.add(key)
        if row['nfe'] not in budgets:
            raise ValueError('Budget not in plan')
        if row.get('quality_pass') is not None and type(row['quality_pass']) is not bool:
            raise ValueError('Quality label must be true, false or null')
        if row.get('matched_context'):
            if row.get('match_errors') or not row.get('anchor_state_hash') or not row.get('noise_hash'):
                raise ValueError('Matched context requires recorded state/noise hashes without match errors')
            if type(row.get('seed')) is not int or not row.get('checkpoint_id') or not row.get('conditioning_id'):
                raise ValueError('Matched context requires pinned generation provenance')
        if row.get('generation_config') is not None and stable_hash(row['generation_config']) != row.get('experiment_id'):
            raise ValueError('Experiment identity does not match recorded generation configuration')
        if row.get('quality_pass') is not None and row.get('label_protocol') is None:
            raise ValueError('Observed labels require protocol version')
        by_comparison.setdefault(row['comparison_id'], []).append(row)
    keys = ('prompt_id', 'prompt_family_id', 'speaker_id', 'text', 'patch_index', 'seed',
            'checkpoint_id', 'conditioning_id', 'reference_nfe', 'experiment_id',
            'generation_config', 'solver', 'preceding_nfe_schedule', 'reference_audio_sha256')
    for cid, rows in by_comparison.items():
        first = rows[0]
        for row in rows[1:]:
            if any(row.get(k) != first.get(k) for k in keys):
                raise ValueError(f'comparison {cid} contains mixed provenance')
        matched = [row for row in rows if row.get('matched_context')]
        if matched:
            for row in matched[1:]:
                if (row['anchor_state_hash'] != matched[0]['anchor_state_hash']
                        or row['noise_hash'] != matched[0]['noise_hash']):
                    raise ValueError(f'comparison {cid} has mismatched starting state/noise')


def dump_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.with_name(path.name + '.tmp')
    with target.open('w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')
    target.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
