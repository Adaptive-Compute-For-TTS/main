"""CPU unit tests. No GPU or installed dots.tts weights are required."""
from __future__ import annotations

import json
import csv
import hashlib
import importlib.util
from copy import deepcopy
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.pilot import (
    PilotPlan, build_observation, comparison_id, dump_jsonl, load_jsonl,
    matching_prefix, select_patch_indices, stable_hash, validate_observations,
)
from training.pilot_labels import QualityPolicy, provisional_label
from training.pilot_capture import tensor_sha256, combined_anchor_hash


def sample_run(*, target=-1, budget=8, reference_nfe=32):
    patches = []
    steps = []
    for i in range(4):
        nfe = budget if i == target else reference_nfe
        steps.append(nfe)
        patches.append({'anchor_state_hash': f'anchor-{i}' if i <= target else f'anchor-{i}-{budget}',
                        'noise_hash': f'noise-{i}',
                        'patch_output_hash': f'patch-{i}' if i < target else f'patch-{i}-{nfe}'})
    return {'reference_nfe': reference_nfe, 'nfe_per_patch': steps, 'patches': patches,
            'audio_path': '/not/generated.wav', 'audio_sha256': 'dummy'}


def test_config_and_patch_positions():
    plan = PilotPlan.from_json(ROOT / 'configs/patch_pilot_v1.json')
    assert len(plan.prompt_ids) == 12
    assert plan.budgets == (4, 8, 16, 32)
    assert select_patch_indices(1) == (0,)
    assert select_patch_indices(2) == (0, 1)
    assert select_patch_indices(10) == (0, 4, 9)
    assert len(select_patch_indices(27)) == 3


def test_invalid_plans():
    with pytest.raises(ValueError):
        PilotPlan(('p001',), (42,), (32, 8), 32, 'speaker')
    with pytest.raises(ValueError):
        select_patch_indices(0)


def test_matched_prefix_and_target_noise():
    target = 2
    baseline, candidate = sample_run(target=target, budget=32), sample_run(target=target, budget=8)
    # Before patch 2 everything is equal; patch 2's output may differ.
    assert matching_prefix(baseline, candidate, target) == []
    candidate['patches'][target]['noise_hash'] = 'other-noise'
    assert 'target_noise_hash_differs' in matching_prefix(baseline, candidate, target)
    candidate['patches'][target]['noise_hash'] = baseline['patches'][target]['noise_hash']
    candidate['patches'][target - 1]['patch_output_hash'] = 'diverged'
    assert any('preceding_patch_1' in err for err in matching_prefix(baseline, candidate, target))


def test_target_not_generated_is_unknown():
    baseline = sample_run(target=2, budget=32)
    candidate = {'reference_nfe':32, 'nfe_per_patch': [], 'patches':[]}
    obs = build_observation(prompt={'id': 'p001', 'text': 'Hello!'}, seed=42, speaker_id='a',
                            patch_index=2, budget=8, candidate=candidate, reference=baseline,
                            checkpoint_id='ck', conditioning_id='cond')
    assert not obs['matched_context']
    assert obs['quality_pass'] is None
    assert obs['match_errors'] == ['target_patch_not_generated']


def test_observation_provenance_and_group_checks(tmp_path):
    ref = sample_run(target=2, budget=32)
    low = sample_run(target=2, budget=8)
    base = dict(prompt={'id':'p001', 'text':'A sentence.', 'prompt_family_id':'f1'},
                seed=42, speaker_id='spk', patch_index=2, reference=ref,
                checkpoint_id='model@sha', conditioning_id='audio@sha')
    rows = [build_observation(**base, budget=b, candidate=c) for b, c in [(8, low), (32, ref)]]
    assert rows[0]['comparison_id'] == rows[1]['comparison_id']
    assert rows[0]['matched_context']
    validate_observations(rows, (8, 32))
    path = tmp_path / 'manifest.jsonl'
    dump_jsonl(path, rows)
    assert load_jsonl(path) == rows
    with pytest.raises(ValueError, match='duplicate'):
        validate_observations(rows + [rows[0]], (8, 32))
    with pytest.raises(ValueError, match='cannot have a quality label'):
        validate_observations([{**rows[0], 'matched_context': False, 'quality_pass': True}], (8, 32))


def test_unknown_and_provisional_label_behaviour():
    pol = QualityPolicy(0.20, 0.10, 0.05, 0.03, 0.1, 0.7)
    row = {'matched_context': True}
    v = {'ok': True}
    reference = {'wer': 0.02, 'cer': 0.01, 'sim_ref': 0.93}
    candidate = {'wer': 0.03, 'cer': 0.02, 'sim_ref': 0.90}
    assert provisional_label(row, v, v, candidate, reference, pol) == (None, 'requires_paired_listening_review')
    assert provisional_label(row, v, v, candidate, reference, pol, True)[0] is True
    assert provisional_label(row, v, v, candidate, reference, pol, False)[0] is False
    assert provisional_label({**row, 'matched_context':False}, v, v, candidate, reference, pol, True)[0] is None
    assert provisional_label(row, {'ok': False}, v, None, reference, pol)[0] is None
    assert provisional_label(row, v, {'ok':False}, candidate, reference, pol)[0] is None
    assert provisional_label(row, v, v, {**candidate,'sim_ref':None}, reference, pol, True)[0] is None
    assert provisional_label(row, v, v, candidate, {**reference, 'wer':0.5}, pol, True)[0] is None


def test_auto_labels_explicit_only():
    pol = QualityPolicy(0.2, 0.1, 0.05, 0.05, require_listening_review=False)
    result = provisional_label({'matched_context':True},{'ok':True},{'ok':True},
                               {'wer':0.01, 'cer':0.01},{'wer':0.,'cer':0.},pol)
    assert result == (True, 'provisional_pass')
    fail = provisional_label({'matched_context':True},{'ok':True},{'ok':True},
                             {'wer':0.35, 'cer':0.11},{'wer':0.,'cer':0.},pol)
    assert fail == (False, 'provisional_fail')


def test_tensor_hash_works_bfloat16():
    t = torch.tensor([[1., 2.]], dtype=torch.bfloat16)
    assert tensor_sha256(t) == tensor_sha256(t.clone())
    assert tensor_sha256(t) != tensor_sha256(t + 1)
    seq = torch.ones((1, 6, 4), dtype=torch.bfloat16)
    cfg = torch.ones_like(seq)
    cond = torch.zeros((1,4), dtype=torch.bfloat16)
    x = combined_anchor_hash(seq,cfg,2,cond)
    seq[:,2:,:] = 7.0  # future positions must not affect the causal prefix hash
    assert combined_anchor_hash(seq,cfg,2,cond) == x
    seq[:,1,:] = 9.0
    assert combined_anchor_hash(seq,cfg,2,cond) != x


def test_hooked_generator_with_fake_eager_runtime(monkeypatch):
    """Exercise the actual patched-runtime hook flow without downloading model weights."""
    import types
    import sys as _sys
    from training.pilot_capture import generate_with_hooks

    class PatchInfo:
        def __init__(self, patch_index, default_nfe, llm_hidden):
            self.patch_index, self.default_nfe, self.llm_hidden = patch_index, default_nfe, llm_hidden

    class Scheduler:
        def __init__(self):
            self.history = []
        def next_nfe(self, *, patch_index, default_nfe, llm_hidden):
            n = self.choose_nfe(PatchInfo(patch_index, default_nfe, llm_hidden))
            self.history.append({'patch':patch_index, 'nfe':n})
            return n

    class FakeRunner:
        latent_patch_size = 2
        latent_dim = 2
        def _prepare_inputs(self, **kw):
            return (kw['sequence'][:, :kw['fm_seq_len']], kw['cfg_sequence'], None, None)
        def _flow_matching_velocity(self, t, z, **kw):
            return 0.1 * z + (0.1 + t) * torch.ones_like(z)

    class Executor:
        def __init__(self, model):
            self.runner = FakeRunner()

    def euler_integrate(velocity, z, n):
        for i in range(n):
            t = torch.tensor(i / n, dtype=z.dtype)
            z = z + (1 / n) * velocity(t, z)
        return z

    pkg = types.ModuleType('adaptive_nfe')
    pkg.__path__ = []
    sched = types.ModuleType('adaptive_nfe.scheduler')
    sched.Scheduler = Scheduler
    sched.PatchInfo = PatchInfo
    executor = types.ModuleType('adaptive_nfe.executor')
    executor.Executor = Executor
    ode = types.ModuleType('adaptive_nfe.ode')
    ode.euler_integrate = euler_integrate
    for module in [pkg, sched, executor, ode]:
        monkeypatch.setitem(_sys.modules, module.__name__, module)

    class FakeModel:
        _optimize_enabled = False
        nfe_scheduler = None
        dit_executor = None

    class FakeRuntime:
        model = FakeModel()
        max_generate_length = 500
        def resolve_sampling_options(self, *, ode_method, num_steps, guidance_scale):
            return ode_method, num_steps, guidance_scale
        def generate(self, *, text, prompt_audio_path, prompt_text, ode_method,
                     num_steps, guidance_scale, speaker_scale):
            patches = []
            # Three sequential patches. Every earlier latent modifies later context.
            state = torch.zeros((1, 4, 4))
            for i in range(3):
                n = self.model.nfe_scheduler.next_nfe(
                    patch_index=i, default_nfe=num_steps, llm_hidden=state[:, i:i+1])
                out = self.model.dit_executor.decode_patch(
                    sequence=state, cfg_sequence=state, fm_seq_len=i+1,
                    g_cond=torch.zeros(1, 4), nfe=n, guidance_scale=guidance_scale)
                patches.append(out.reshape(-1))
                state[:, i+1, :] = out.flatten()[:4]
            return {'audio': torch.cat(patches), 'sample_rate': 16, 'time_used':0.01}

    rt = FakeRuntime()
    common = dict(text='Hi', seed=42, reference_nfe=8, prompt_audio_path=None,
                  speaker_scale=1.5, guidance_scale=1.2)
    a = generate_with_hooks(rt, target_patch=None, target_nfe=None, **common)
    b = generate_with_hooks(rt, target_patch=1, target_nfe=4, **common)
    assert a['nfe_per_patch'] == [8,8,8]
    assert b['nfe_per_patch'] == [8,4,8]
    assert matching_prefix(a,b,1) == []
    assert a['patches'][1]['patch_output_hash'] != b['patches'][1]['patch_output_hash']
    assert rt.model.nfe_scheduler is None
    assert rt.model.dit_executor is None
    # Early EOS is auditable, not a fabricated failure or a discarded run.
    missing = generate_with_hooks(rt, target_patch=9, target_nfe=4, **common)
    assert matching_prefix(a, missing, 9) == ['target_patch_not_generated']

    def broken_generation(**kwargs):
        raise RuntimeError('synthetic runtime error')
    monkeypatch.setattr(rt, 'generate', broken_generation)
    with pytest.raises(RuntimeError, match='synthetic runtime error'):
        generate_with_hooks(rt, target_patch=None, target_nfe=None, **common)
    assert rt.model.nfe_scheduler is None and rt.model.dit_executor is None


def load_script(name):
    spec = importlib.util.spec_from_file_location(f'pilot_script_{name}', ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_windows_safe_root_dry_run(tmp_path):
    import os
    import subprocess
    env = dict(os.environ, PYTHONIOENCODING='cp1252')
    run = subprocess.run([sys.executable, str(ROOT / 'scripts/generate_patch_pilot.py'),
                          '--dry-run', '--out-dir', str(tmp_path / 'no-output')],
                         cwd=ROOT.parent, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert 'up to 288 observations' in run.stdout
    assert '240 maximum' in run.stdout
    assert not (tmp_path / 'no-output').exists()


def test_pinned_checkpoint_is_passed_to_loader_and_verified():
    generator = load_script('generate_patch_pilot')
    revision = 'a' * 40  # synthetic commit for a loader spy, not real model weights
    from types import SimpleNamespace
    calls = []

    class RuntimeSpy:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            calls.append((model, kwargs))
            return SimpleNamespace(pretrained_path=Path('/synthetic/snapshots') / kwargs['revision'])

    generator.load_pinned_runtime(RuntimeSpy, model='unit/test', checkpoint_id=f'unit/test@{revision}',
                                  precision='bfloat16', max_generate_length=500)
    assert calls[0][1]['revision'] == revision
    assert calls[0][1]['optimize'] is False
    for invalid in ('unit/test@main', 'unit/other@' + revision, 'metadata-only-tag'):
        with pytest.raises(ValueError, match='FULL_40_HEX_COMMIT'):
            generator.load_pinned_runtime(RuntimeSpy, model='unit/test', checkpoint_id=invalid,
                                          precision='bfloat16', max_generate_length=500)
    assert len(calls) == 1


@pytest.mark.parametrize('mismatch', [None, 'audio', 'sample_rate', 'schedule'])
def test_mandatory_equivalence_compares_audio_rate_and_schedule(monkeypatch, mismatch):
    import types
    generator = load_script('generate_patch_pilot')
    audio = torch.linspace(-0.1, 0.1, 32)
    reference = {'audio_sha256': hashlib.sha256(audio.numpy().tobytes()).hexdigest(),
                 'sample_rate': 16000, 'nfe_per_patch': [32, 32]}
    canonical = types.SimpleNamespace(output={'audio': audio.clone(), 'sample_rate': 16000},
                                      nfe_per_patch=[32, 32])
    if mismatch == 'audio':
        canonical.output['audio'] += 0.01
    elif mismatch == 'sample_rate':
        canonical.output['sample_rate'] = 8000
    elif mismatch == 'schedule':
        canonical.nfe_per_patch = [4, 4]
    package = types.ModuleType('adaptive_nfe')
    package.generate_fixed_nfe = lambda *args, **kwargs: canonical
    monkeypatch.setitem(sys.modules, 'adaptive_nfe', package)
    options = dict(seed=42, text='Synthetic fixture.', reference_nfe=32,
                   ref_audio=None, guidance_scale=1.2, speaker_scale=1.5)
    if mismatch:
        with pytest.raises(RuntimeError, match='changed fixed-NFE audio'):
            generator.verify_reference_equivalence(None, reference, **options)
    else:
        assert generator.verify_reference_equivalence(None, reference, **options)['passed'] is True
    with pytest.raises(SystemExit):
        generator.main(['--dry-run', '--skip-recorder-equivalence'])


def test_unmatched_first_row_does_not_hide_mismatched_matched_rows():
    ref = sample_run(target=1, budget=32)
    rows = [build_observation(prompt={'id': 'test', 'text': 'Synthetic.'}, seed=1, speaker_id='test',
                              patch_index=1, budget=n, candidate=sample_run(target=1, budget=n), reference=ref,
                              checkpoint_id='synthetic', conditioning_id='synthetic') for n in (4, 8, 32)]
    rows[0].update(matched_context=False, match_errors=['synthetic-mismatch'])
    rows[1]['noise_hash'] = 'mismatched-noise'
    with pytest.raises(ValueError, match='mismatched starting state/noise'):
        validate_observations(rows, (4, 8, 32))


@pytest.mark.parametrize('mutation', ['missing_hash', 'non_target_nfe', 'wrong_target_budget'])
def test_incomplete_or_wrong_interventions_are_not_matched(mutation):
    ref, candidate = sample_run(target=1, budget=32), sample_run(target=1, budget=4)
    if mutation == 'missing_hash':
        ref['patches'][1]['noise_hash'] = candidate['patches'][1]['noise_hash'] = None
    elif mutation == 'non_target_nfe':
        candidate['nfe_per_patch'][3] = 8
    else:
        candidate['nfe_per_patch'][1] = 8
    row = build_observation(prompt={'id': 'test', 'text': 'Synthetic.'}, seed=1, speaker_id='test',
                            patch_index=1, budget=4, candidate=candidate, reference=ref,
                            checkpoint_id='synthetic', conditioning_id='synthetic')
    assert not row['matched_context'] and row['quality_pass'] is None


@pytest.mark.parametrize('bad', [None, float('nan'), float('inf'), -0.1, True])
def test_missing_or_invalid_metrics_stay_unknown(bad):
    policy = QualityPolicy(0.2, 0.1, 0.05, 0.03, require_listening_review=False)
    result = provisional_label({'matched_context': True}, {'ok': True}, {'ok': True},
                               {'wer': bad, 'cer': 0.01}, {'wer': 0.01, 'cer': 0.01}, policy)
    assert result[0] is None


def test_truncated_and_bad_reference_voice_stay_unknown():
    policy = QualityPolicy(0.2, 0.1, 0.05, 0.03, min_speaker_sim=0.7)
    scores = {'wer': 0.01, 'cer': 0.01, 'sim_ref': 0.9}
    assert provisional_label({'matched_context': True}, {'ok': True, 'flags': ['truncated_at_cap']},
                             {'ok': True}, scores, scores, policy, True)[0] is None
    assert provisional_label({'matched_context': True}, {'ok': True}, {'ok': True}, scores,
                             {**scores, 'sim_ref': 0.1}, policy, True)[0] is None
    with pytest.raises(ValueError, match='Explicit WER/CER thresholds'):
        QualityPolicy(None, 0.1, 0.05, 0.03)


def test_prompt_snapshot_matches_repository():
    snapshot = json.loads((ROOT / 'configs/patch_pilot_prompts_v1.json').read_text(encoding='utf-8'))
    benchmark = {r['id']: r for r in json.loads((ROOT.parent / 'quality-evals/wer_cer/prompts/benchmark_prompts_v1.json').read_text(encoding='utf-8'))}
    for row in snapshot:
        assert all(row[key] == benchmark[row['id']][key] for key in ('text', 'category', 'length_bucket'))


@pytest.fixture
def scoring_fixture(tmp_path, monkeypatch):
    """Real WAV/scorer interfaces; ASR and speaker predictions are synthetic test doubles."""
    np = pytest.importorskip('numpy')
    sf = pytest.importorskip('soundfile')
    pytest.importorskip('wav_validation')
    for dependency in ('faster_whisper', 'jiwer', 'whisper_normalizer', 'pandas'):
        pytest.importorskip(dependency)
    monkeypatch.syspath_prepend(str(ROOT.parent / 'quality-evals/speaker_sim'))
    import evaluate_run
    real_evaluator = evaluate_run.QualityEvaluator
    text = 'A synthetic test sentence.'

    class SyntheticSpeaker:
        def similarity(self, a, b):
            return 0.9  # Unit-test fixture only, not a measured speaker score.

    monkeypatch.setattr(evaluate_run, 'QualityEvaluator', lambda reference_wav=None: real_evaluator(
        reference_wav=reference_wav, transcribe=lambda path: text, sim=SyntheticSpeaker()))
    audio = (0.2 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)).astype('float32')
    wav = tmp_path / 'synthetic-test-only.wav'
    sf.write(wav, audio, 16000, subtype='FLOAT')
    sha = hashlib.sha256(sf.read(wav, dtype='float32')[0].tobytes()).hexdigest()
    experiment = {'ref_audio_sha256': hashlib.sha256(wav.read_bytes()).hexdigest(), 'purpose': 'unit-test-only'}
    ref, candidate = sample_run(target=1, budget=32), sample_run(target=1, budget=4)
    for run in (ref, candidate):
        run.update(audio_path=str(wav), audio_sha256=sha, sample_rate=16000, max_generate_length=500)
    rows = [build_observation(prompt={'id': 'unit-test', 'text': text}, seed=42, speaker_id='test',
                              patch_index=1, budget=b, candidate=run, reference=ref,
                              checkpoint_id='synthetic-test-only', conditioning_id='test',
                              experiment=experiment, recorder_equivalence=True)
            for b, run in ((4, candidate), (32, ref))]
    source = tmp_path / 'observations.unscored.jsonl'
    dump_jsonl(source, rows)
    scorer = load_script('score_patch_pilot')
    args = ['--input', str(source), '--max-wer', '0.2', '--max-cer', '0.1',
            '--max-wer-delta', '0.05', '--max-cer-delta', '0.03',
            '--speaker-ref', str(wav), '--min-sim', '0.7']
    return scorer, args, rows, tmp_path


def test_scoring_review_and_approved_v2_handoff(scoring_fixture):
    scorer, args, rows, directory = scoring_fixture
    assert scorer.main(args) == 0
    scored = load_jsonl(directory / 'observations.provisional.jsonl')
    assert all(row['quality_pass'] is None for row in scored)
    assert all(row['candidate_validation']['ok'] and row['candidate_quality']['wer'] == 0 for row in scored)
    assert all(row['candidate_quality']['sim_ref'] == 0.9 for row in scored)
    assert load_jsonl(directory / 'observations.label_ready.jsonl') == []
    review_path = directory / 'listening_review_template.csv'
    with review_path.open(newline='') as fp:
        reader = csv.DictReader(fp)
        fields, reviews = reader.fieldnames, list(reader)
    for review in reviews:
        review['listening_pass'] = 'pass'  # Synthetic fixture review, not real listening approval.
    with review_path.open('w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reviews)
    assert scorer.main(args + ['--reviews', str(review_path)]) == 0
    reviewed = load_jsonl(directory / 'observations.label_ready.jsonl')
    assert len(reviewed) == 2 and all(row['quality_pass'] is True for row in reviewed)
    from training.pilot_labels import approved_training_observations
    from training.targets import build_targets
    from training.train import prepare_data
    approved = approved_training_observations(reviewed, approved_protocol_version='synthetic-test-approved-v1')
    target = build_targets(approved)[0]
    assert target.schema_version == 'patch-nfe-v2'
    assert target.mask == (1, 0, 0, 0, 0, 0, 0, 0, 1)
    # Confirm the additive adapter actually reaches the existing training preparation.
    expanded = []
    for i in range(3):
        for row in deepcopy(approved):
            row.update(comparison_id=f'unit-comparison-{i}', prompt_id=f'p{i}', prompt_family_id=f'family{i}')
            expanded.append(row)
    assert set(prepare_data(build_targets(expanded)).datasets) == {'train', 'val', 'test'}
    with pytest.raises(ValueError, match='approved protocol'):
        approved_training_observations(reviewed, approved_protocol_version='unapproved-v0')
    # A stale review of another audio pair must never turn into a failed label.
    reviews[0]['candidate_audio_sha256'] = 'a-different-audio-pair'
    with review_path.open('w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reviews)
    scorer.main(args + ['--reviews', str(review_path)])
    assert load_jsonl(directory / 'observations.provisional.jsonl')[0]['quality_pass'] is None


def test_auto_scores_are_not_label_ready_and_changed_audio_stays_unknown(scoring_fixture):
    scorer, args, rows, directory = scoring_fixture
    scorer.main(args + ['--auto-provisional'])
    scored = load_jsonl(directory / 'observations.provisional.jsonl')
    assert all(row['quality_pass'] is True for row in scored)
    assert load_jsonl(directory / 'observations.label_ready.jsonl') == []
    from training.pilot_labels import approved_training_observations
    with pytest.raises(ValueError, match='listening review'):
        approved_training_observations(scored, approved_protocol_version='synthetic-approved-v1')
    rows[0]['candidate_audio_sha256'] = 'changed-samples'
    dump_jsonl(directory / 'observations.unscored.jsonl', rows)
    scorer.main(args + ['--auto-provisional'])
    changed = load_jsonl(directory / 'observations.provisional.jsonl')[0]
    assert changed['quality_pass'] is None and changed['label_reason'] == 'audio_integrity_unverified'


def test_wav_cache_rejects_changed_samples_settings_and_sample_rate(tmp_path, monkeypatch):
    sf = pytest.importorskip('soundfile')
    import numpy as np
    import training.pilot_capture as capture
    generator = load_script('generate_patch_pilot')
    audio = np.linspace(-0.1, 0.1, 160, dtype='float32')
    result = dict(sample_run(target=1, budget=4), audio=audio, sample_rate=16000,
                  audio_sha256=hashlib.sha256(audio.tobytes()).hexdigest())
    monkeypatch.setattr(capture, 'generate_with_hooks', lambda *args, **kwargs: result)
    options = dict(run_config={'synthetic': True}, text='Test', seed=42, reference_nfe=32,
                   target_patch=1, target_nfe=4, ref_audio=None, speaker_scale=1.5, guidance_scale=1.2)
    first = generator.save_run(None, tmp_path, **options)
    assert generator.save_run(None, tmp_path, **options) == first
    with pytest.raises(RuntimeError, match='different settings'):
        generator.save_run(None, tmp_path, **{**options, 'run_config': {'synthetic': False}})
    sf.write(tmp_path / 'audio.wav', audio, 8000, subtype='FLOAT')
    with pytest.raises(RuntimeError, match='WAV hash mismatch'):
        generator.save_run(None, tmp_path, **options)
