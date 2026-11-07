# wav-validation

Checks whether a generated WAV is usable. Owner: Nick Kim.

Status: **imported, not yet refactored.** The files here are copied unchanged from
`nicholaskim17/watai_exp` (`experiments/clean_samples/`, commit `7555334`). The next step is
turning the batch script into a reusable `validate_wav()` that the GPU worker can call on each
`audio.wav` (see `gpu-workers/README.md`).

## What's here

| File | What it is |
| --- | --- |
| `flag_wav_runs.py` | Flags every archived NFE benchmark WAV as `clean_candidate`, `truncated_at_cap` or `contaminated_continuation`, with signal checks. Batch script; paths are specific to the watai_exp archive. |
| `wav_flags.csv` | Its output: one row per archived WAV (1,028 rows). `quality-evals/speaker_sim/build_run_json.py --flags` reads this file. |
| `run_summary.json` | Per-NFE and per-length-bucket medians for the two full runs. |

## Flags in `wav_flags.csv`

- `contaminated_continuation`: the run passed the reference transcript as continuation context, so the reference intro can leak into the output.
- `truncated_at_cap`: stopped at `max_generate_length` (80 patches = 12.8 s), usually mid-sentence.
- `clean_candidate`: x-vector conditioning and ended before the cap.

`signal_ok` additionally requires no clipping, peak <= 0.95, leading silence < 0.5 s, trailing silence >= 0.05 s and no internal gap > 1.0 s.

## Rerunning the imported script

It only works inside the watai_exp repo, which has `experiments/nfe_benchmark_runs/` and `experiments/benchmark_prompts_v1.csv`:

```bash
python experiments/clean_samples/flag_wav_runs.py
```

Requires `numpy` and `soundfile`.
