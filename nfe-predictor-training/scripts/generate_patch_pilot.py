"""Collect controlled single-patch NFE variations on a machine with dots.tts GPU setup.

No model weights or GPU are needed for --dry-run. This is NOT quality labeling.
Run from the monorepo root or nfe-predictor-training directory:

    python nfe-predictor-training/scripts/generate_patch_pilot.py --dry-run
    python nfe-predictor-training/scripts/generate_patch_pilot.py \\
        --checkpoint-id dots-ttss-soar-pinned-REVISION \\
        --ref-audio /path/to/reference.wav --limit-prompts 1 --limit-seeds 1

The script rejects cache mismatches rather than silently combining runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from training.pilot import (PilotPlan, build_observation, dump_jsonl, load_jsonl,
                            select_patch_indices, stable_hash, validate_observations)

PROMPTS_DEFAULT = ROOT / 'quality-evals' / 'wer_cer' / 'prompts' / 'benchmark_prompts_v1.json'
CONFIG_DEFAULT = HERE / 'configs' / 'patch_pilot_v1.json'


def load_pinned_runtime(runtime_class, *, model, checkpoint_id, precision, max_generate_length):
    """Bind the recorded ID to the snapshot actually loaded, not just metadata."""
    prefix, separator, revision = checkpoint_id.rpartition('@')
    if separator != '@' or prefix != model or not re.fullmatch(r'[0-9a-fA-F]{40}', revision):
        raise ValueError('--checkpoint-id must be MODEL_REPO@FULL_40_HEX_COMMIT matching --model')
    if Path(model).exists():
        raise ValueError('Use the model repository ID plus its immutable revision, not an unchecked local directory')
    runtime = runtime_class.from_pretrained(
        model, revision=revision.lower(), precision=precision, optimize=False,
        max_generate_length=max_generate_length)
    if Path(runtime.pretrained_path).name.lower() != revision.lower():
        raise RuntimeError('Resolved model snapshot does not match the recorded checkpoint revision')
    return runtime


def verify_reference_equivalence(runtime, reference, *, seed, text, reference_nfe,
                                 ref_audio, guidance_scale, speaker_scale):
    """Mandatory check against the team's unchanged fixed-NFE Executor."""
    from adaptive_nfe import generate_fixed_nfe
    from training.pilot_capture import seed_all
    seed_all(seed)
    original = generate_fixed_nfe(
        runtime, nfe=reference_nfe, text=text, prompt_audio_path=ref_audio,
        prompt_text=None, ode_method='euler', guidance_scale=guidance_scale,
        speaker_scale=speaker_scale)
    canonical_audio = original.output['audio'].detach().float().cpu().squeeze().numpy()
    canonical_hash = hashlib.sha256(canonical_audio.tobytes()).hexdigest()
    if (canonical_hash != reference['audio_sha256']
            or int(original.output['sample_rate']) != reference['sample_rate']
            or original.nfe_per_patch != reference['nfe_per_patch']):
        raise RuntimeError('Recording executor changed fixed-NFE audio; stop the dataset run')
    return {'passed': True, 'canonical_audio_sha256': canonical_hash,
            'recorder_audio_sha256': reference['audio_sha256'],
            'sample_rate': reference['sample_rate'], 'nfe_per_patch': original.nfe_per_patch,
            'seed': seed}


def file_sha(path: Path | None) -> str | None:
    if path is None:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prompts(path: Path, prompt_ids: tuple[str, ...]) -> list[dict]:
    raw = json.loads(path.read_text(encoding='utf-8'))
    lookup = {x['id']: x for x in raw}
    if len(lookup) != len(raw):
        raise ValueError('Duplicate prompt IDs in input')
    missing = sorted(set(prompt_ids) - set(lookup))
    if missing:
        raise ValueError(f'Prompts missing from manifest: {missing}')
    return [lookup[k] for k in prompt_ids]


def save_run(runtime, run_dir: Path, *, run_config: dict, text: str,
             seed: int, reference_nfe: int, target_patch: int | None, target_nfe: int | None,
             ref_audio: str | None, speaker_scale: float, guidance_scale: float):
    """Use completed cache only if provenance matches; store full per-patch trace."""
    if ref_audio and file_sha(Path(ref_audio)) != run_config.get('ref_audio_sha256'):
        raise RuntimeError('Speaker reference changed after experiment identity was recorded')
    summary_file, wav_file = run_dir / 'trace.json', run_dir / 'audio.wav'
    if summary_file.exists() or wav_file.exists():
        if not summary_file.is_file() or not wav_file.is_file():
            raise RuntimeError(f'Incomplete cached run at {run_dir}; inspect/remove it before retrying')
        summary = json.loads(summary_file.read_text(encoding='utf-8'))
        if summary.get('run_config_hash') != stable_hash(run_config):
            raise RuntimeError(f'Cached run has different settings: {run_dir}')
        import soundfile as sf
        audio, sample_rate = sf.read(wav_file, dtype='float32')
        if (audio.ndim != 1 or sample_rate != summary['sample_rate']
                or hashlib.sha256(audio.tobytes()).hexdigest() != summary['audio_sha256']):
            raise RuntimeError(f'Cached WAV hash mismatch: {wav_file}')
        return summary
    from training.pilot_capture import generate_with_hooks
    import soundfile as sf
    result = generate_with_hooks(
        runtime, text=text, seed=seed, reference_nfe=reference_nfe,
        target_patch=target_patch, target_nfe=target_nfe,
        prompt_audio_path=ref_audio, speaker_scale=speaker_scale,
        guidance_scale=guidance_scale)
    run_dir.mkdir(parents=True, exist_ok=True)
    # The run trace is written last: an interrupted write won't be resumed as complete.
    sf.write(wav_file, result['audio'], result['sample_rate'], subtype='FLOAT')
    output = {key: value for key, value in result.items() if key != 'audio'}
    output.update(audio_path=str(wav_file.resolve()), run_config_hash=stable_hash(run_config),
                  num_patches=len(result['nfe_per_patch']))
    # Hash on the samples re-read from the WAV (some backends alter sample representation).
    audio_check, _ = sf.read(wav_file, dtype='float32')
    if hashlib.sha256(audio_check.tobytes()).hexdigest() != result['audio_sha256']:
        raise RuntimeError('Float WAV roundtrip did not preserve the sample hash')
    summary_file.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    return output


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=CONFIG_DEFAULT)
    p.add_argument('--prompts', type=Path, default=PROMPTS_DEFAULT)
    p.add_argument('--out-dir', type=Path, default=HERE / 'runs' / 'patch_pilot_v1')
    p.add_argument('--checkpoint-id', help='Exact model repo/revision or snapshot hash for reproducibility')
    p.add_argument('--model', default='dots-studio/dots.tts-soar')
    p.add_argument('--ref-audio', type=Path, help='Existing speaker reference WAV for x-vector cloning')
    p.add_argument('--precision', default='bfloat16')
    p.add_argument('--guidance-scale', type=float, default=1.2)
    p.add_argument('--speaker-scale', type=float, default=1.5)
    p.add_argument('--limit-prompts', type=int, default=None)
    p.add_argument('--limit-seeds', type=int, default=None)
    p.add_argument('--budgets', nargs='+', type=int, help='Optional small subset for smoke run, must include reference')
    p.add_argument('--target-patches', type=int, default=None)
    p.add_argument('--max-generate-length', type=int, default=500)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    plan = PilotPlan.from_json(a.config)
    budgets = tuple(a.budgets) if a.budgets is not None else plan.budgets
    if a.limit_prompts is not None and a.limit_prompts < 1 or a.limit_seeds is not None and a.limit_seeds < 1:
        p.error('--limit-prompts and --limit-seeds must be >= 1')
    prompts = load_prompts(a.prompts, plan.prompt_ids[:a.limit_prompts])
    seeds = plan.seeds[:a.limit_seeds]
    target_patches = a.target_patches if a.target_patches is not None else plan.target_patches
    if a.max_generate_length < 100:
        p.error('--max-generate-length must be >= 100 to avoid a short generation cap')
    PilotPlan(tuple(pr['id'] for pr in prompts), seeds, budgets, plan.reference_nfe, plan.speaker_id, target_patches)
    ref = a.ref_audio.resolve() if a.ref_audio else None
    if ref and not ref.is_file():
        p.error(f'Reference audio not found: {ref}')
    conditioning_id = stable_hash({'ref_audio_sha256': file_sha(ref), 'prompt_text': None,
                                   'guidance_scale': a.guidance_scale, 'speaker_scale': a.speaker_scale})
    identity = {'model': a.model, 'checkpoint_id': a.checkpoint_id, 'precision': a.precision,
                'optimize': False, 'ode_method': 'euler', 'ref_audio_sha256': file_sha(ref),
                'conditioning_id': conditioning_id, 'speaker_id': plan.speaker_id,
                'budgets': budgets, 'reference_nfe': plan.reference_nfe,
                'seeds': seeds, 'target_patches': target_patches,
                'prompt_ids': [pr['id'] for pr in prompts],
                'guidance_scale': a.guidance_scale, 'speaker_scale': a.speaker_scale,
                'max_generate_length': a.max_generate_length,
                'collector_sha256': {name: file_sha(HERE / name) for name in (
                    'training/pilot.py', 'training/pilot_capture.py', 'scripts/generate_patch_pilot.py')},
                'prompt_manifest_sha256': file_sha(a.prompts),
                'config_file_sha256': file_sha(a.config)}
    expected = len(prompts) * len(seeds) * target_patches * len(budgets)
    print(f'Plan: {len(prompts)} prompts x {len(seeds)} seeds x up to {target_patches} targets x {len(budgets)} budgets = up to {expected} observations')
    print(f'Collection generations: {len(prompts) * len(seeds) * (1 + target_patches * (len(budgets) - 1))} maximum, plus 1 mandatory equivalence generation')
    if a.dry_run:
        print('DRY RUN: no model loaded, no WAVs or quality labels generated')
        return 0
    if not a.checkpoint_id or any(mark in a.checkpoint_id.upper() for mark in ('TBD', 'PIN_EXACT', 'PLACEHOLDER', 'UNVERIFIED')):
        p.error('--checkpoint-id must record the pinned model revision/checkpoint')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Real pilot requires a GPU with patched dots.tts. No CPU fallback is implied.')
    from dots_tts.runtime import DotsTtsRuntime
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rt = load_pinned_runtime(DotsTtsRuntime, model=a.model, checkpoint_id=a.checkpoint_id,
                             precision=a.precision, max_generate_length=a.max_generate_length)
    if getattr(rt, 'max_generate_length', None) is not None and rt.max_generate_length < 100:
        raise RuntimeError('Generation cap is too low; configure dots.tts for sufficiently long audio first')
    import inspect
    from adaptive_nfe.executor import Executor
    from adaptive_nfe.ode import euler_integrate
    identity.update(
        resolved_snapshot=str(rt.pretrained_path), torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        deterministic_algorithms=True, max_sequence_length=rt.max_sequence_length,
        vocoder_merge_steps=rt.vocoder_merge_steps,
        runtime_code_sha256=file_sha(Path(inspect.getfile(DotsTtsRuntime))),
        model_code_sha256=file_sha(Path(inspect.getfile(type(rt.model)))),
        executor_code_sha256=file_sha(Path(inspect.getfile(Executor))),
        ode_code_sha256=file_sha(Path(inspect.getfile(euler_integrate))))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    config_path = a.out_dir / 'experiment.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding='utf-8'))
        if previous['identity'] != json.loads(json.dumps(identity)):
            raise RuntimeError('Output dir already belongs to a different experiment')
    else:
        config_path.write_text(json.dumps({'schema_version': 'patch-nfe-v1', 'identity': identity,
                                           'torch': torch.__version__,
                                           'cuda': torch.version.cuda,
                                           'gpu': torch.cuda.get_device_name(0)}, indent=2), encoding='utf-8')
    manifest: list[dict] = []
    first_ref_checked = False
    for prompt in prompts:
        for seed in seeds:
            base_dir = a.out_dir / prompt['id'] / f'seed_{seed}'
            common_config = {**identity, 'prompt_id': prompt['id'], 'text': prompt['text'], 'seed': seed}
            reference = save_run(rt, base_dir / 'reference', run_config={**common_config, 'target_patch': None,
                                    'target_nfe': None}, text=prompt['text'], seed=seed,
                                 reference_nfe=plan.reference_nfe, target_patch=None, target_nfe=None,
                                 ref_audio=str(ref) if ref else None,
                                 speaker_scale=a.speaker_scale, guidance_scale=a.guidance_scale)
            if not first_ref_checked:
                evidence = verify_reference_equivalence(
                    rt, reference, seed=seed, text=prompt['text'], reference_nfe=plan.reference_nfe,
                    ref_audio=str(ref) if ref else None, guidance_scale=a.guidance_scale,
                    speaker_scale=a.speaker_scale)
                evidence.update(experiment_id=stable_hash(identity), prompt_id=prompt['id'])
                (a.out_dir / 'equivalence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
                print('Recorder reference matches canonical fixed-NFE Executor output (SHA256).')
                first_ref_checked = True
            indices = select_patch_indices(len(reference['nfe_per_patch']), target_patches)
            for patch_idx in indices:
                for budget in budgets:
                    if budget == plan.reference_nfe:
                        candidate = reference
                    else:
                        cfg = {**common_config, 'target_patch': patch_idx, 'target_nfe': budget}
                        candidate = save_run(rt, base_dir / f'patch_{patch_idx}' / f'nfe_{budget}',
                                             run_config=cfg, text=prompt['text'], seed=seed,
                                             reference_nfe=plan.reference_nfe, target_patch=patch_idx,
                                             target_nfe=budget, ref_audio=str(ref) if ref else None,
                                             speaker_scale=a.speaker_scale, guidance_scale=a.guidance_scale)
                    row = build_observation(prompt=prompt, seed=seed, speaker_id=plan.speaker_id,
                                            patch_index=patch_idx, budget=budget, candidate=candidate,
                                            reference=reference, checkpoint_id=a.checkpoint_id,
                                            conditioning_id=conditioning_id, experiment=identity,
                                            recorder_equivalence=first_ref_checked)
                    manifest.append(row)
                    validate_observations(manifest, budgets)
                    dump_jsonl(a.out_dir / 'observations.unscored.jsonl', manifest)
                    print(f'{prompt["id"]} s{seed} patch={patch_idx} nfe={budget}: matched={row["matched_context"]} errors={row["match_errors"]}')
    print(f'Wrote {len(manifest)} unscored observations -> {a.out_dir / "observations.unscored.jsonl"}')
    print('Next: score_patch_pilot.py; generated observations have null quality labels by design.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
