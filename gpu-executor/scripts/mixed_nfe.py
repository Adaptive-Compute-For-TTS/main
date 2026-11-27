"""Mixed per-patch NFE on the real model: does it run, is the audio valid, and what does switching cost?

Runs one prompt under several schedules (fixed and mixed) on one path, with per-patch DiT timing:

    python scripts/mixed_nfe.py                       # eager
    python scripts/mixed_nfe.py --optimize            # optimized path (KV cache, compiled steps; ~5 min warmup)
    python scripts/mixed_nfe.py --out-dir runs/mixed --text "Some sentence."

On the optimized path, a patch whose NFE differs from the previous patch's rebuilds the DiT KV cache
(K/V for the whole history, at every ODE step). The "switch_ms" column shows that cost.

Writes <out-dir>/<path>/<schedule>.wav and <out-dir>/<path>/summary.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import soundfile as sf
import torch
from adaptive_nfe import ListScheduler, generate_fixed_nfe, generate_scheduled
from adaptive_nfe.executor import Executor
from dots_tts.runtime import DotsTtsRuntime

TEXT = ("The scheduler decides how many denoising steps each patch of audio receives, "
        "so easy parts of a sentence can be cheaper than hard ones.")

SCHEDULES = {
    "fixed32": [32],
    "fixed16": [16],
    "fixed8": [8],
    "high_then_8": [32, 32, 32, 8],             # 1 switch: hard start, cheap rest
    "blocks_8_16": [8] * 6 + [16] * 6 + [8],    # 2 switches
    "alternate_8_32": [8, 32] * 40,             # a switch every patch: the worst case for the KV cache
}


def audio_sha256(audio: torch.Tensor) -> str:
    return hashlib.sha256(audio.float().cpu().contiguous().numpy().tobytes()).hexdigest()


def seed_all(s: int) -> None:
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="dots-studio/dots.tts-soar")
    p.add_argument("--optimize", action="store_true")
    p.add_argument("--text", default=TEXT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance-scale", type=float, default=1.2)
    p.add_argument("--out-dir", type=Path, default=Path("runs/mixed_nfe"))
    a = p.parse_args()

    path = "optimized" if a.optimize else "eager"
    out = a.out_dir / path
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rt = DotsTtsRuntime.from_pretrained(a.model, precision="bfloat16", optimize=a.optimize)
    ex = Executor(rt.model)
    kw = dict(text=a.text, guidance_scale=a.guidance_scale)
    print(f"[{path}] loaded in {time.perf_counter() - t0:.1f}s")

    # Warm up every NFE used (compiles / captures graphs on the optimized path), plus a switch.
    t0 = time.perf_counter()
    for nfe in (8, 16, 32):
        seed_all(a.seed)
        generate_fixed_nfe(rt, nfe, executor=ex, text="Warm up.", guidance_scale=a.guidance_scale)
    seed_all(a.seed)
    generate_scheduled(rt, ListScheduler([8, 32, 16]), executor=ex, text="Warm up please.",
                       guidance_scale=a.guidance_scale)
    print(f"[{path}] warmup {time.perf_counter() - t0:.1f}s")

    # Sanity: a constant schedule through generate_scheduled == generate_fixed_nfe, bit for bit.
    seed_all(a.seed)
    ref = generate_fixed_nfe(rt, 16, executor=ex, **kw)
    seed_all(a.seed)
    same = generate_scheduled(rt, ListScheduler([16]), executor=ex, **kw)
    identical = audio_sha256(ref.output["audio"]) == audio_sha256(same.output["audio"])
    print(f"[{path}] constant schedule matches generate_fixed_nfe: {identical}")

    rows = []
    for name, sched in SCHEDULES.items():
        seed_all(a.seed)
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        r = generate_scheduled(rt, ListScheduler(sched), executor=ex, time_patches=True, **kw)
        wall = time.perf_counter() - t
        audio, sr = r.output["audio"], int(r.output["sample_rate"])
        sf.write(out / f"{name}.wav", audio.float().cpu().numpy().reshape(-1), sr)
        a_np = audio.float().cpu()
        # Per-patch cost: ms per DiT step, split by "NFE changed from the previous patch" or not.
        per_step = [ms / n for ms, n in zip(r.patch_ms, r.nfe_per_patch)]
        switched = [i for i in range(1, r.num_patches) if r.nfe_per_patch[i] != r.nfe_per_patch[i - 1]]
        steady = [i for i in range(1, r.num_patches) if i not in switched]
        row = {
            "schedule": name, "patches": r.num_patches, "nfe_per_patch": r.nfe_per_patch,
            "switches": r.nfe_switches, "total_dit_steps": r.total_dit_steps,
            "mean_nfe": round(r.total_dit_steps / r.num_patches, 2),
            "wall_s": round(wall, 3), "dit_ms": round(sum(r.patch_ms), 1),
            "audio_s": round(a_np.numel() / sr, 3), "rtf": round(wall / (a_np.numel() / sr), 3),
            "ms_per_step_steady": round(statistics.median([per_step[i] for i in steady]), 3) if steady else None,
            "ms_per_step_switch": round(statistics.median([per_step[i] for i in switched]), 3) if switched else None,
            "switch_ms": [round(r.patch_ms[i], 1) for i in switched],
            "patch_ms": [round(x, 1) for x in r.patch_ms],
            "finite": bool(torch.isfinite(a_np).all()), "peak_abs": round(float(a_np.abs().max()), 4),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20),
            "audio_sha256": audio_sha256(audio),
        }
        rows.append(row)
        print(f"[{path}] {name:15s} patches={row['patches']:3d} switches={row['switches']:3d} "
              f"steps={row['total_dit_steps']:4d} wall={row['wall_s']:6.2f}s dit={row['dit_ms']:7.1f}ms "
              f"ms/step steady={row['ms_per_step_steady']} switch={row['ms_per_step_switch']} "
              f"audio={row['audio_s']}s finite={row['finite']}")

    summary = {"path": path, "model": a.model, "text": a.text, "seed": a.seed,
               "constant_schedule_matches_fixed": identical, "runs": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{path}] wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
