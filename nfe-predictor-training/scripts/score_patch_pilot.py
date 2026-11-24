"""Score controlled patch interventions using the team's existing WAV/quality scorers.

All thresholds are USER-SUPPLIED AND UNAPPROVED. Default behavior emits no
pass/fail labels until paired listening reviews exist. --auto-provisional is an
explicit experimental override and NEVER means final quality approval.

Requires installs from wav-validation/ and quality-evals/speaker_sim/.
"""
from __future__ import annotations

import argparse
import csv
import json
import hashlib
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from training.pilot import dump_jsonl, load_jsonl, stable_hash, validate_observations
from training.pilot_labels import QualityPolicy, approved_training_observations, provisional_label

REVIEW_IDENTITY = ('experiment_id', 'candidate_audio_sha256', 'reference_audio_sha256',
                   'anchor_state_hash', 'noise_hash')


def read_reviews(path: Path | None):
    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    reviews = {}
    with path.open(encoding='utf-8', newline='') as fp:
        for row in csv.DictReader(fp):
            key = (row['comparison_id'], int(row['nfe']))
            if key in reviews:
                raise ValueError(f'Duplicate manual review: {key}')
            s = row['listening_pass'].strip().lower()
            if s in ('pass', 'true', 'yes', '1'):
                verdict = True
            elif s in ('fail', 'false', 'no', '0'):
                verdict = False
            elif s in ('', 'unknown', 'unsure'):
                verdict = None
            else:
                raise ValueError(f'Unrecognized review for {key}: {s}')
            reviews[key] = {**{k: row.get(k) for k in REVIEW_IDENTITY},
                            'listening_pass': verdict, 'notes': row.get('notes', '')}
    return reviews


def write_review_template(rows, path: Path):
    """A simple listening queue, one row per candidate; never invent verdicts."""
    with path.open('w', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=('comparison_id', 'nfe', 'prompt_id', 'patch_index',
                                                 'candidate_audio_path', 'reference_audio_path',
                                                 *REVIEW_IDENTITY,
                                                 'listening_pass', 'notes'))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='observations.unscored.jsonl')
    p.add_argument('--output', type=Path, help='Defaults to observations.provisional.jsonl next to input')
    p.add_argument('--speaker-ref', type=Path, help="Reference WAV for the team's speaker-sim scorer")
    p.add_argument('--reviews', type=Path, help='Optional CSV with listening_pass for each candidate')
    p.add_argument('--max-wer', type=float, required=True, help='Absolute WER ceiling, explicitly provisional')
    p.add_argument('--max-cer', type=float, required=True, help='Absolute CER ceiling, explicitly provisional')
    p.add_argument('--max-wer-delta', type=float, required=True, help='Allowed increase relative to fixed reference')
    p.add_argument('--max-cer-delta', type=float, required=True)
    p.add_argument('--max-sim-drop', type=float, help='Optional voice similarity degradation tolerance')
    p.add_argument('--min-sim', type=float, help='Optional absolute speaker similarity floor')
    p.add_argument('--auto-provisional', action='store_true', help='Allow auto-only EXPLORATORY labels without human review')
    p.add_argument('--approved-protocol-version', help='Explicit team approval attestation for a reviewed v2 training export; never use for exploratory labels')
    a = p.parse_args(argv)
    rows = load_jsonl(a.input)
    if not rows:
        raise RuntimeError('Empty input manifest')
    validate_observations(rows, tuple(sorted({row['nfe'] for row in rows})))
    if a.output and a.output.resolve() == a.input.resolve():
        p.error('--output must not overwrite the unscored input manifest')
    if a.approved_protocol_version and a.auto_provisional:
        p.error('Approved training export cannot use --auto-provisional')
    review_template = a.input.with_name('listening_review_template.csv')
    if not review_template.exists():
        write_review_template(rows, review_template)
    reviews = read_reviews(a.reviews)
    if (a.min_sim is not None or a.max_sim_drop is not None) and a.speaker_ref is None:
        p.error('--speaker-ref is required for speaker-sim thresholds')
    if a.speaker_ref:
        if not a.speaker_ref.is_file():
            p.error('--speaker-ref must be an existing WAV')
        reference_hash = hashlib.sha256(a.speaker_ref.read_bytes()).hexdigest()
        if any((row.get('generation_config') or {}).get('ref_audio_sha256') != reference_hash for row in rows):
            p.error('--speaker-ref must match the reference audio hash recorded during generation')
    pol = QualityPolicy(a.max_wer, a.max_cer, a.max_wer_delta, a.max_cer_delta,
                        a.max_sim_drop, a.min_sim, not a.auto_provisional,
                        'pilot-auto-unapproved-v0' if a.auto_provisional else 'pilot-listening-unapproved-v0')
    try:
        from wav_validation import validate_wav
    except ImportError as e:
        raise RuntimeError('Install wav-validation: pip install -e wav-validation') from e
    try:
        from evaluate_run import QualityEvaluator
    except ImportError:
        # Shared implementation in the monorepo, no copied scorer logic.
        sys.path.insert(0, str(HERE.parent / 'quality-evals' / 'speaker_sim'))
        try:
            from evaluate_run import QualityEvaluator
        except ImportError as e:
            raise RuntimeError('Install quality-evals/speaker_sim/requirements-worker.txt with gpu-executor/constraints.txt; see PATCH_PILOT_README.md') from e
    evaluator = QualityEvaluator(reference_wav=str(a.speaker_ref) if a.speaker_ref else None)
    cache = {}

    def score_one(audio_path, prompt, n_patches, max_patches, expected_sha, expected_rate):
        if not audio_path:
            return None, None, False
        key = (audio_path, prompt, n_patches, max_patches, expected_sha, expected_rate)
        if key not in cache:
            validation = validate_wav(audio_path, num_patches=n_patches, max_generate_length=max_patches)
            verified = False
            if Path(audio_path).is_file() and expected_sha:
                import soundfile as sf
                try:
                    audio, rate = sf.read(audio_path, dtype='float32')
                    verified = (audio.ndim == 1 and expected_rate == rate
                                and hashlib.sha256(audio.tobytes()).hexdigest() == expected_sha)
                except (OSError, RuntimeError, ValueError):
                    pass
            quality = evaluator.evaluate(audio_path, prompt, validation=validation) if verified and validation.get('ok') else None
            if quality:
                quality = dict(quality)
                for metric in ('wer', 'cer', 'sim_ref'):
                    value = quality.get(metric)
                    if value is not None and (type(value) not in (float, int) or not math.isfinite(value)):
                        quality[metric] = None
            cache[key] = validation, quality, verified
        return cache[key]

    for row in rows:
        r_val, r_quality, r_verified = score_one(row.get('reference_audio_path'), row['text'],
            row.get('reference_num_patches'), row.get('reference_max_generate_length'),
            row.get('reference_audio_sha256'), row.get('reference_sample_rate'))
        c_val, c_quality, c_verified = score_one(row.get('candidate_audio_path'), row['text'],
            row.get('candidate_num_patches'), row.get('candidate_max_generate_length'),
            row.get('candidate_audio_sha256'), row.get('candidate_sample_rate'))
        key = (row['comparison_id'], row['nfe'])
        review = reviews.get(key, {})
        label, reason = provisional_label(row, c_val, r_val, c_quality, r_quality, pol,
                                          listening_pass=review.get('listening_pass'))
        review_verified = type(review.get('listening_pass')) is bool and all(
            row.get(k) and review.get(k) == row[k] for k in REVIEW_IDENTITY)
        if not r_verified or not c_verified:
            label, reason = None, 'audio_integrity_unverified'
        elif review and type(review.get('listening_pass')) is bool and not review_verified:
            label, reason = None, 'review_provenance_mismatch'
        elif row.get('recorder_equivalence') is not True:
            label, reason = None, 'recorder_equivalence_unverified'
        row['candidate_validation'], row['reference_validation'] = c_val, r_val
        row['candidate_quality'], row['reference_quality'] = c_quality, r_quality
        row['quality_pass'] = label
        row['label_protocol'] = pol.protocol_version if label is not None else None
        row['label_reason'] = reason
        row['manual_review'] = review if review else None
        row['review_verified'] = bool(review_verified)
        row['audio_integrity_verified'] = bool(r_verified and c_verified)
        row['label_policy'] = vars(pol)
        row['label_policy_sha256'] = stable_hash(vars(pol))
    validate_observations(rows, tuple(sorted({row['nfe'] for row in rows})))
    output = a.output or a.input.with_name('observations.provisional.jsonl')
    dump_jsonl(output, rows)
    # Existing training.targets.build_targets demands all NFE candidates in each group
    # have a matched anchor. Invalid-context observations must be excluded.
    training_rows = [row for row in rows if row['matched_context'] and row['quality_pass'] is not None
                     and row['review_verified']]
    training_output = output.with_name('observations.label_ready.jsonl')
    dump_jsonl(training_output, training_rows)
    report = {'total_candidates': len(rows), 'matched': sum(r['matched_context'] for r in rows),
              'labelled_provisional': sum(r['quality_pass'] is not None for r in rows),
              'reviewed_provisional': len(training_rows),
              'passes': sum(r['quality_pass'] is True for r in rows),
              'failures': sum(r['quality_pass'] is False for r in rows),
              'unknown': sum(r['quality_pass'] is None for r in rows),
              'label_protocol': pol.protocol_version, 'team_approved': False,
              'policy': vars(pol), 'manual_reviews_count': len(reviews)}
    if a.approved_protocol_version:
        approved = approved_training_observations(rows, approved_protocol_version=a.approved_protocol_version)
        dump_jsonl(output.with_name('observations.training_v2.jsonl'), approved)
        report['training_export_protocol'] = a.approved_protocol_version
        report['training_export_observations'] = len(approved)
        report['team_approved'] = True
    output.with_suffix('.report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('Full score output:', output)
    print('Filtered label-ready output:', training_output)
    print('Listening queue:', review_template)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
