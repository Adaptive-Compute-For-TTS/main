# Mixed per-patch NFE: what switching costs

Run on 2026-11-26: `scripts/mixed_nfe.py`, dots.tts-soar, bf16, one ~7.4 s sentence, seed 42, RTX 5060 Laptop
(8 GB), WSL2. Per-patch DiT time is CUDA-synchronized (`time_patches=True`), which adds a sync per patch.
Raw numbers: `results/mixed_nfe/{eager,optimized}_soar_rtx5060.json`.

## Results

| Schedule | NFE switches | DiT steps | Eager wall | Optimized wall |
|---|---:|---:|---:|---:|
| fixed 32 | 0 | ~1450 | 26.1 s | 17.0 s |
| fixed 16 | 0 | 736 | 14.3 s | 7.5 s |
| fixed 8 | 0 | 376 | 8.8 s | 4.2 s |
| 32, 32, 32, then 8 | 1 | 448 | 8.9 s | 5.6 s |
| blocks: 8×6, 16×6, then 8 | 2 | ~420 | 9.0 s | 5.8 s |
| alternate 8 / 32 every patch | ~46 | ~940 | 20.8 s | **54.5 s** |

All runs produced finite audio. A constant schedule through `generate_scheduled` is bit-identical to
`generate_fixed_nfe` on both paths. Patch counts differ by a few between schedules because a different
NFE changes the generated audio and so where EOS lands.

### Optimized path, DiT time per patch

| Patch | Same NFE as the previous patch | NFE changed |
|---|---:|---:|
| at NFE 8 | ~49 ms | ~420-470 ms |
| at NFE 16 | ~124 ms | ~970 ms |
| at NFE 32 | ~307 ms | ~1900 ms |

## Why

- **Eager:** no DiT KV cache, every step recomputes attention over the whole sequence. A step costs about
  16 ms whatever the schedule, so time follows the number of DiT steps and switching is free.
- **Optimized:** the DiT KV cache keeps one K/V copy per ODE step (`cache_k[i]`). The K/V at step i depends
  on the adaLN modulation for time t_i = i/nfe, so a cache only fits one NFE. On a change, dots.tts
  (`DiTSolver._ensure_cache`) allocates a new cache and `runner.prefill` recomputes K/V for the whole
  history at every ODE step. Correct, but each switch costs about 8× a normal patch. In this short
  utterance it barely grew with position (1890 → 1985 ms at NFE 32 over 46 patches).

## What it means for adaptive NFE

- On the optimized path, a schedule that flips NFE often is **slower than fixed 32** (54.5 s vs 17.0 s).
  Savings come from long runs at a low NFE.
- Break-even on this GPU: dropping 32 → 8 costs ~420 ms once and saves ~258 ms per patch after, so it pays
  off after about 2 patches. Going back up 8 → 32 costs ~1600 ms extra.
- The policy should therefore be switch-aware: for example a minimum run length, hysteresis, or a switch
  penalty in the decision. The model's per-patch probabilities stay the same; only the selection changes.

## Executor ideas to make switching cheap (not built)

- **Nested time grids.** With t_i = i/nfe, the times of NFE 8 are every 4th time of NFE 32
  (i/8 = 4i/32), and NFE 16 is every 2nd. One cache kept at the finest grid (32) could serve 8 and 16 by
  striding, if every patch's K/V were written for all 32 times.
- **Incremental catch-up.** A finished patch's K/V at a given time depends only on that patch and t_i, not
  on the noise. So instead of re-prefilling the whole history, compute only the missing times for the patches
  decoded since the cache was last used. The cost then depends on how many patches were missed, not on the
  utterance length.
- Both mean changing how the optimized path writes and reads the cache (`optimized.py`, and dots.tts's
  `_ensure_cache`). Worth doing only if the trained predictor actually switches often.
