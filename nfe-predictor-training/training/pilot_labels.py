"""Conservative, explicitly provisional patch label policy.

Automated WER/CER or speaker similarity for a *whole utterance* can be used as
screening evidence for one controlled patch intervention, but is NOT a direct
measurement of perceptual quality in a 160-ms patch. Human review is required
for final labels; auto-only labels remain tagged provisional.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class QualityPolicy:
    max_absolute_wer: float
    max_absolute_cer: float
    max_delta_wer: float
    max_delta_cer: float
    max_speaker_sim_drop: float | None = None
    min_speaker_sim: float | None = None
    require_listening_review: bool = True
    protocol_version: str = 'pilot-unapproved-v0'

    def __post_init__(self):
        if any(value is None for value in (self.max_absolute_wer, self.max_absolute_cer,
                                          self.max_delta_wer, self.max_delta_cer)):
            raise ValueError('Explicit WER/CER thresholds are required')
        values = (self.max_absolute_wer, self.max_absolute_cer, self.max_delta_wer,
                  self.max_delta_cer, self.max_speaker_sim_drop, self.min_speaker_sim)
        if any(x is not None and (type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1) for x in values):
            raise ValueError('All quality thresholds must be between 0 and 1')
        if type(self.require_listening_review) is not bool:
            raise ValueError('require_listening_review must be boolean')
        if not isinstance(self.protocol_version, str) or not self.protocol_version.strip():
            raise ValueError('label protocol version is required')


def provisional_label(row: Mapping[str, Any], candidate_val: Mapping[str, Any] | None,
                      reference_val: Mapping[str, Any] | None,
                      candidate: Mapping[str, Any] | None,
                      reference: Mapping[str, Any] | None,
                      policy: QualityPolicy,
                      listening_pass: bool | None = None) -> tuple[bool | None, str]:
    if row.get('matched_context') is not True or row.get('match_errors'):
        return None, 'unmatched_context'
    if reference_val is None or not reference_val.get('ok'):
        return None, 'reference_invalid'
    if 'truncated_at_cap' in reference_val.get('flags', []) or reference_val.get('checks', {}).get('at_cap'):
        return None, 'reference_truncated'
    if candidate_val is None:
        return None, 'candidate_unvalidated'
    if not candidate_val.get('ok'):
        return None, 'candidate_invalid_or_unavailable'
    if 'truncated_at_cap' in candidate_val.get('flags', []) or candidate_val.get('checks', {}).get('at_cap'):
        return None, 'candidate_truncated'
    if not reference or not candidate:
        return None, 'quality_metrics_missing'
    for key in ('wer', 'cer'):
        if any(type(scores.get(key)) not in (int, float) or not math.isfinite(scores[key])
               or not 0 <= scores[key] <= 1 for scores in (candidate, reference)):
            return None, f'{key}_missing'
    if reference['wer'] > policy.max_absolute_wer or reference['cer'] > policy.max_absolute_cer:
        return None, 'reference_fails_absolute_intelligibility_bar'
    speaker_active = policy.max_speaker_sim_drop is not None or policy.min_speaker_sim is not None
    if speaker_active:
        if any(type(scores.get('sim_ref')) not in (int, float) or not math.isfinite(scores['sim_ref'])
               or not -1 <= scores['sim_ref'] <= 1 for scores in (candidate, reference)):
            return None, 'speaker_sim_missing'
        if policy.min_speaker_sim is not None and reference['sim_ref'] < policy.min_speaker_sim:
            return None, 'reference_fails_speaker_bar'
    good = (candidate['wer'] <= policy.max_absolute_wer and
            candidate['cer'] <= policy.max_absolute_cer and
            candidate['wer'] - reference['wer'] <= policy.max_delta_wer and
            candidate['cer'] - reference['cer'] <= policy.max_delta_cer)
    if policy.max_speaker_sim_drop is not None:
        good &= reference['sim_ref'] - candidate['sim_ref'] <= policy.max_speaker_sim_drop
    if policy.min_speaker_sim is not None:
        good &= candidate['sim_ref'] >= policy.min_speaker_sim
    if listening_pass is False:
        good = False
    if policy.require_listening_review and listening_pass is None:
        return None, 'requires_paired_listening_review'
    return bool(good), 'provisional_pass' if good else 'provisional_fail'


def approved_training_observations(rows, *, approved_protocol_version: str):
    """Explicit human-approved v1 -> v2 handoff; never approve an auto-only label.

    The caller attests that the team approved the protocol. Listening decisions
    must be bound to the exact audio pair by score_patch_pilot.py. No old training
    contracts or model dimensions are changed. Unknown outcomes stay unknown.
    """
    from training.contracts import CANDIDATE_NFES, EXPERIMENT_SCHEMA_VERSION, FEATURE_SCHEMA_VERSION, validate_observation
    from training.pilot import stable_hash, validate_observations

    if (not isinstance(approved_protocol_version, str) or not approved_protocol_version.strip()
            or any(x in approved_protocol_version.lower() for x in ('unapproved', 'pending', 'tbd'))):
        raise ValueError('An explicitly approved protocol version is required')
    validate_observations(rows, CANDIDATE_NFES)
    output = []
    for row in rows:
        if not row.get('matched_context'):
            continue
        if row.get('recorder_equivalence') is not True or not row.get('experiment_id'):
            raise ValueError('Training export requires recorded executor equivalence and experiment identity')
        label = row.get('quality_pass')
        review = row.get('manual_review') or {}
        if label is not None:
            if (type(review.get('listening_pass')) is not bool or not row.get('review_verified')
                    or row.get('audio_integrity_verified') is not True):
                raise ValueError('Training export requires an audio-bound paired listening review')
            policy = row.get('label_policy')
            if not policy or row.get('label_policy_sha256') != stable_hash(policy):
                raise ValueError('Training export requires the recorded scoring policy')
            expected, _ = provisional_label(
                row, row.get('candidate_validation'), row.get('reference_validation'),
                row.get('candidate_quality'), row.get('reference_quality'),
                QualityPolicy(**{**policy, 'require_listening_review': True}), review['listening_pass'])
            if expected is None or expected != label:
                raise ValueError('Provisional label does not match reviewed quality evidence')
        result = deepcopy(row)
        val = row.get('candidate_validation') or {}
        result.update(
            schema_version=EXPERIMENT_SCHEMA_VERSION, feature_schema_version=FEATURE_SCHEMA_VERSION,
            record_status='measured', patch_id=f"{row['prompt_id']}:patch-{row['patch_index']}",
            solver='euler', source_label_protocol=row.get('label_protocol'),
            label_protocol=approved_protocol_version if label is not None else None,
            audio_valid=bool(val['ok']) if 'ok' in val else None,
            not_truncated=('truncated_at_cap' not in val.get('flags', [])
                           and not val.get('checks', {}).get('at_cap')) if val else None,
            counterfactual_reviewed=type(review.get('listening_pass')) is bool,
            quality_measurements={f'{side}_{key}': (
                (row.get(f'{side}_quality') or {}).get(key)
                if type((row.get(f'{side}_quality') or {}).get(key)) in (int, float)
                and math.isfinite((row.get(f'{side}_quality') or {})[key]) else None)
                                  for side in ('candidate', 'reference') for key in ('wer', 'cer', 'sim_ref')},
        )
        validate_observation(result, CANDIDATE_NFES)
        output.append(result)
    return output
