"""Opt-in eager dots.tts single-patch experiment recorder.

Uses the team's existing scheduler and eager DiT runner. It does not modify
production executor code or monkeypatch global torch functions. The recorder
owns exactly the same noise draw and Euler update as Executor.decode_patch;
a separate equivalence check on the target GPU is recommended before research.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any

import torch


def tensor_sha256(value: torch.Tensor | None) -> str | None:
    if value is None:
        return None
    t = value.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def combined_anchor_hash(sequence: torch.Tensor, cfg_sequence: torch.Tensor | None,
                         fm_seq_len: int, g_cond: torch.Tensor) -> str:
    """Hash the actual causal conditioning prefix and voice conditioning."""
    if fm_seq_len < 1:
        raise ValueError('fm_seq_len must be >= 1')
    metadata = {
        'fm_seq_len': fm_seq_len,
        'sequence': tensor_sha256(sequence[:, :fm_seq_len]),
        'cfg_sequence': tensor_sha256(cfg_sequence[:, :fm_seq_len]) if cfg_sequence is not None else None,
        'g_cond': tensor_sha256(g_cond),
    }
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()


def seed_all(seed: int) -> None:
    import numpy as np
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_pilot_hooks(model: Any, reference_nfe: int, target_patch: int | None,
                      target_nfe: int | None):
    """Return (scheduler, executor) using the installed patched dots.tts components."""
    try:
        from adaptive_nfe.scheduler import Scheduler, PatchInfo
        from adaptive_nfe.executor import Executor
        from adaptive_nfe.ode import euler_integrate
    except ImportError as exc:
        raise RuntimeError('Install gpu-executor and patched dots.tts per gpu-executor/README.md') from exc

    class PilotScheduler(Scheduler):
        def __init__(self):
            super().__init__()
            self.llm_hidden_by_patch: dict[int, str | None] = {}

        def choose_nfe(self, info: PatchInfo) -> int:
            self.llm_hidden_by_patch[info.patch_index] = tensor_sha256(info.llm_hidden)
            if target_patch is not None and info.patch_index == target_patch:
                return int(target_nfe)
            return reference_nfe

    scheduler = PilotScheduler()

    class RecordingExecutor(Executor):
        def __init__(self, core):
            super().__init__(core)
            self.patches: list[dict[str, Any]] = []

        @torch.no_grad()
        def decode_patch(self, *, sequence: torch.Tensor, cfg_sequence: torch.Tensor | None,
                         fm_seq_len: int, g_cond: torch.Tensor, nfe: int,
                         guidance_scale: float, ode_method: str = 'euler') -> torch.Tensor:
            if ode_method != 'euler':
                raise NotImplementedError('Pilot recorder supports Euler in eager mode only')
            if nfe < 1 or fm_seq_len < 1:
                raise ValueError('Invalid pilot nfe or fm_seq_len')
            if not scheduler.history:
                raise RuntimeError('The patched dots.tts scheduler hook was not invoked')
            patch_index = scheduler.history[-1]['patch']
            if len(self.patches) != patch_index:
                raise RuntimeError(f'Unexpected patch numbering: {patch_index} after {len(self.patches)} calls')
            anchor_hash = combined_anchor_hash(sequence, cfg_sequence, fm_seq_len, g_cond)
            runner = self.runner
            inp, cfg_inp, mask, pos = runner._prepare_inputs(
                sequence=sequence, cfg_sequence=cfg_sequence, fm_seq_len=fm_seq_len)
            guidance = sequence.new_tensor(float(guidance_scale))
            # Exactly the same sampled noise call as gpu-executor.Executor.decode_patch.
            noise = torch.randn(
                (sequence.size(0), runner.latent_patch_size, runner.latent_dim),
                dtype=sequence.dtype, device=sequence.device)

            def velocity(t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
                return runner._flow_matching_velocity(
                    t, z, input_sequence=inp, cfg_sequence=cfg_inp,
                    attn_mask=mask, pos_ids=pos, g_cond=g_cond, guidance_scale=guidance)

            patch = euler_integrate(velocity, noise, nfe)
            self.patches.append({
                'patch_index': patch_index,
                'nfe': nfe,
                'anchor_state_hash': anchor_hash,
                'noise_hash': tensor_sha256(noise),
                'patch_output_hash': tensor_sha256(patch),
                'llm_hidden_hash': scheduler.llm_hidden_by_patch.get(patch_index),
            })
            return patch

    return scheduler, RecordingExecutor(model)


def generate_with_hooks(runtime: Any, *, text: str, seed: int, reference_nfe: int,
                        target_patch: int | None, target_nfe: int | None,
                        prompt_audio_path: str | None, speaker_scale: float,
                        guidance_scale: float) -> dict[str, Any]:
    """Return waveform and per-patch hashes; always restore installed hooks."""
    if getattr(runtime.model, '_optimize_enabled', False):
        raise ValueError('Pilot requires optimize=False; optimized path has variable-NFE cache overhead')
    if target_patch is not None and (target_patch < 0 or target_nfe is None):
        raise ValueError('A target patch requires non-negative index and candidate NFE')
    scheduler, executor = build_pilot_hooks(runtime.model, reference_nfe, target_patch, target_nfe)
    ode, steps, cfg = runtime.resolve_sampling_options(
        ode_method='euler', num_steps=reference_nfe, guidance_scale=guidance_scale)
    if ode != 'euler' or int(steps) != reference_nfe:
        raise RuntimeError('Runtime did not resolve the requested reference Euler configuration')
    seed_all(seed)
    saved = (getattr(runtime.model, 'nfe_scheduler', None), getattr(runtime.model, 'dit_executor', None))
    runtime.model.nfe_scheduler = scheduler
    runtime.model.dit_executor = executor
    try:
        result = runtime.generate(
            text=text, prompt_audio_path=prompt_audio_path, prompt_text=None,
            ode_method=ode, num_steps=steps, guidance_scale=cfg, speaker_scale=speaker_scale)
    finally:
        runtime.model.nfe_scheduler, runtime.model.dit_executor = saved
    nfe_per_patch = [entry['nfe'] for entry in scheduler.history]
    if len(nfe_per_patch) != len(executor.patches) or not nfe_per_patch:
        raise RuntimeError('Scheduler and executor patch records disagree')
    # Retain early-EOS audio/trace for audit; matching_prefix marks the target unknown.
    for i, nfe in enumerate(nfe_per_patch):
        if nfe != (target_nfe if i == target_patch else reference_nfe):
            raise RuntimeError('Pilot schedule was not applied')
    import numpy as np
    audio = result['audio'].detach().float().cpu().squeeze().numpy()
    if audio.ndim != 1:
        raise RuntimeError(f'Expected mono audio, got {audio.shape}')
    sr = int(result['sample_rate'])
    return {
        'audio': audio,
        'audio_sha256': hashlib.sha256(audio.tobytes()).hexdigest(),
        'sample_rate': sr,
        'audio_s': len(audio) / sr,
        'time_s': float(result.get('time_used', 0)),
        'nfe_per_patch': nfe_per_patch,
        'patches': executor.patches,
        'reference_nfe': reference_nfe,
        'seed': seed,
        'max_generate_length': getattr(runtime, 'max_generate_length', None),
    }
