# Phase 2 results: speaker similarity, WER and latency vs NFE

Data: Nick's two full sweeps in [watai_exp](https://github.com/Adaptive-Compute-For-TTS/watai_exp) (`experiments/nfe_benchmark_runs`), 54 prompts (18 short / 18 medium / 18 long) × NFE 4, 6, 8, 10, 12, 16, 20, 24, 32, seed 42, one generation each, all cloning `experiments/my_files/abhi1.wav`. WER from Arush (Whisper "small", per clip, 2026-10-15).

| run_id | folders | GPU (run owner) |
|---|---|---|
| `xvector_4090` | `nfe_54_prompts_v1_full_xvector_part_0/1` | RTX 4090, x-vector conditioning |
| `orig_3090` | `nfe_54_prompts_v1_full_part_0/1` | RTX 3090, conditioning not recorded |

- **SIM** = WavLM-base-plus-sv cosine similarity (Phase 1 scorer). For scale, from Phase 1: same real person 0.93–0.97, two different people 0.73.
- **WER** = Arush's `wer`: word error rate against the prompt text, capped at 1, with any re-spoken reference intro removed first. **WER on spoken words** (`wer_heard`) = errors ÷ words the clip got to before it was cut off, which separates "said it wrong" from "ran out of time". **Cut off** = at least one word missing at the end.
- **RTF** = generation time ÷ audio length (below 1 = faster than real time).
- CIs are 95% bootstrap over prompts. Full tables: `results/watai_exp/` (`summary_by_nfe.csv`, `paired_deltas.csv`, `latency_fit.csv`, `clips.csv`, `summary.json`); by prompt type in `results/watai_exp/by_content/`.

## Bottom line

On the x-vector run, **NFE 4 says the words as well as NFE 32 and sounds almost as much like the target voice, for 5× less compute, and it's the only setting faster than real time.** The biggest quality problem in these sweeps isn't NFE at all: it's the generation length cap cutting off long prompts.

## Main findings (x-vector 4090 run)

| NFE | 4 | 6 | 8 | 10 | 12 | 16 | 20 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| SIM vs reference | 0.917 | 0.923 | 0.924 | 0.928 | 0.929 | 0.930 | 0.925 | 0.930 | 0.930 |
| WER | 0.121 | 0.135 | 0.119 | 0.113 | 0.108 | 0.109 | 0.129 | 0.116 | 0.108 |
| WER on spoken words | 0.022 | 0.031 | 0.028 | 0.025 | 0.020 | 0.027 | 0.026 | 0.031 | 0.023 |
| Clips cut off | 33% | 35% | 33% | 33% | 35% | 35% | 31% | 33% | 33% |
| RTF | 0.81 | 1.05 | 1.30 | 1.51 | 1.72 | 2.20 | 2.67 | 3.16 | 4.09 |

**1. WER doesn't depend on NFE.** Paired with the same prompt at NFE 32, no NFE from 4 to 24 changes WER on spoken words (every 95% CI includes 0). Overall WER shows one blip (NFE 6: +0.026, CI +0.005 to +0.050) that the spoken-words WER doesn't confirm, so it is most likely truncation noise.

**2. Long prompts get cut off at every NFE.** Every x-vector clip stops at 12.8 s at most, and 157 of the 162 long-prompt clips (about 43 words each) hit that cap with words missing. That's why long prompts have WER ≈ 0.25–0.31 while short and medium sit at 0.02–0.06; on the words they did say, long prompts score 0.01–0.03 like everything else. The cap matches `max_generate_length = 80` in the benchmark settings (80 × 0.16 s = 12.8 s); raising it is a generation-config fix, not an NFE question.

**3. The cloned voice holds up at every NFE.** SIM stays inside the Phase 1 same-person range (0.910–0.976) at every NFE and far above two different people (0.73). Paired with NFE 32, SIM drops a little only below NFE 10:

| NFE | 4 | 6 | 8 | 10 | 12 | 16 | 20 | 24 |
|---|---|---|---|---|---|---|---|---|
| SIM change vs NFE 32 | **−0.013** | **−0.008** | **−0.006** | −0.002 | −0.001 | 0.000 | −0.005* | 0.000 |

Bold = 95% CI excludes 0. (*NFE 20 also excludes 0, but with this many comparisons about one false alarm is expected, and its neighbours show nothing.) Even the NFE 4 drop is less than half the gap between two recordings of the same real person in Phase 1 (0.969, i.e. 0.031 below identical).

**4. Cost is linear in NFE: RTF ≈ 0.34 + 0.117 × NFE** (R² 0.97; intercept CI 0.31–0.38, slope CI 0.115–0.119). Every extra NFE adds about 0.12 s of compute per second of audio, on top of about 0.34 s fixed. **Only NFE 4 is faster than real time (RTF 0.81); NFE 6 is about 1.0.** Raw latency fits look worse (R² 0.38) only because clip lengths differ; per length bucket they're tight (long: 4.0 s + 1.50 s/NFE, R² 0.99).

**5. Short prompts score lower SIM at every NFE** (about 0.90 vs 0.93–0.94 for medium/long), including NFE 32. This is most likely the scorer having less audio (1.5–3 s) to build a voiceprint from, not worse cloning, so compare NFEs within a length bucket rather than across buckets.

**6. Prompt type:** at NFE 4 the SIM drop is clearest for numbers (−0.032) and plain sentences (−0.025). With only 9 prompts per type, the CIs overlap too much to rank the types reliably.

## Valid clips only (what the per-run dashboard JSON shows)

`build_run_json.py` drops invalid clips (see README, "Per-run JSON") before averaging. For `xvector_4090` that leaves the 324 short and medium clips that the generation cap did not cut off (36 prompts per NFE, same 12.8 s cap). The conclusions above hold:

| NFE | 4 | 6 | 8 | 10 | 12 | 16 | 20 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| SIM vs reference | 0.910 | 0.914 | 0.917 | 0.923 | 0.923 | 0.925 | 0.919 | 0.923 | 0.924 |
| WER | 0.025 | 0.045 | 0.044 | 0.037 | 0.035 | 0.038 | 0.040 | 0.040 | 0.026 |
| RTF | 0.83 | 1.05 | 1.32 | 1.52 | 1.72 | 2.20 | 2.67 | 3.17 | 4.12 |

- **SIM:** paired with NFE 32, the drop is −0.014 at NFE 4 and −0.010 at NFE 6 (both 95% CIs exclude 0), and nothing from NFE 8 up is distinguishable from 0.
- **WER:** changes are at most 0.02 and nearly all CIs include 0. One delta (NFE 16, +0.012) excludes 0, one hit in eight comparisons, so don't read it as a trend. Absolute levels (0.025–0.045) are close to the 0.047 Whisper small gave on Arush's human recordings.
- **Cost:** RTF = 0.35 + 0.117 × NFE (R² 0.97), as before.
- Short prompts sit 0.02–0.04 below medium on SIM at every NFE, as before; compare NFEs within a bucket.

## The original 3090 run is not usable for NFE conclusions

- **It mostly says the wrong thing:** WER 0.70–0.82 at every NFE (median 1.0). In 481 of 486 clips the model first re-speaks the reference recording's own text ("Hey, Adam here. Let's create something…"), then runs into the 9.12 s cap before finishing the prompt: 80–91% of clips are cut off, missing about 20 words on average. Only short prompts get through (WER 0.16–0.50).
- **4 generations are silent** (peak level ~0.002, about 50× quieter than speech): `p036` at NFE 6, and `p018`, `p034`, `p045` at NFE 8. Arush's ASR rejected the same 4. The SIM scorer now rejects near-silent audio (peak below 0.01) and counts them as failures (`failed` metric) instead of scoring them.
- Its latencies are noisier (some NFE 6–8 clips took 40–48 s), and it differs from the x-vector run in GPU *and* conditioning, so differences between the runs can't be pinned on either one.

**Recommendation:** use `xvector_4090` as the reference sweep. The re-spoken intro in `orig_3090` looks like the reference recording leaking into generation in that configuration. The numbers fit that: 12.8 s (the x-vector cap) minus the 3.69 s reference clip is 9.11 s, almost exactly the original run's 9.12 s cap. That's an inference, not confirmed; worth flagging to Nick.

## Caveats

- One reference speaker, one seed, one generation per prompt/NFE.
- WER uses Whisper "small" (Arush's pipeline, org repo `tts_eval_org`). Text is lowercased, punctuation stripped and contractions expanded; numbers are compared as written (e.g. `$42.75`), so number prompts may carry some ASR formatting error. On Arush's own human recordings of 3 benchmark prompts, the same setup gave WER 0.047, so the 0.02–0.05 seen on short and medium prompts is close to what the ASR gets on real speech.
- SIM from WavLM-base-plus-sv; not directly comparable with papers that use WavLM-large.

## Reproduce

See the "Real data" commands in `README.md`. All checks: `python sanity_check.py` (Phase 1, 16/16) and `python pipeline_check.py` (Phase 2 and the flagging code, 62/62).
