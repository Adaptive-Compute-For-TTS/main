# wav-validation

Checks every generated clip before it's stored or scored. Owner: Nick Kim.

## Where it sits

In the system diagram, WAV Validation comes right after Local GPU Execution:

```
Local GPU Execution ──► WAV Validation ──► Local WAV Storage ──► audio_path saved to SQLite ──► FastAPI results
                                     └──► Quality Evaluation (only clips that passed)
```

For each run, the GPU worker hands over the audio it just generated. This package checks it:
- **If it passes:** the WAV is saved to storage, the run's `audio_path` points at it, and the clip goes on to quality evaluation.
- **If it fails:** the run is marked failed with the reason. The WAV is kept in a `_rejected/` folder for debugging, but it is never served or scored.

It only needs `numpy` and `soundfile` (no torch), and works on Python 3.9 and later, so the worker and the backend can both import it.

## What it checks

Every flag is either **invalid** (the run fails) or a **caution** (reported, but the run still succeeds).

| flag | severity | when |
|---|---|---|
| `missing_audio` | invalid | the WAV file isn't on disk (file checks only) |
| `unreadable_audio` | invalid | the file can't be decoded (file checks only) |
| `non_finite` | invalid | any NaN or infinite sample (a numerical blow-up; the executor writes float32) |
| `too_short` | invalid | shorter than 0.5 s |
| `silent` | invalid | peak below 0.01 (about -40 dBFS) or RMS below 1e-4 |
| `truncated_at_cap` | invalid | generation hit `max_generate_length`, so the sentence is probably cut off |
| `clipped` | caution | any sample at full scale (abs value ≥ 0.999) |
| `too_loud` | caution | peak above 0.95 |
| `leading_silence` | caution | starts with 0.5 s or more of silence |
| `abrupt_end` | caution | less than 0.05 s of silence at the end |
| `long_gap` | caution | a pause longer than 1.0 s in the middle |

The thresholds are the team's existing numbers, not new ones:
- `too_short` and `silent` are the speaker-similarity scorer's own rules (`quality-evals/speaker_sim/speaker_sim.py`), so a clip that passes here can always be scored.
- The five cautions are the `signal_ok` rules from the benchmark's `flag_wav_runs.py`.
- `tests/test_thresholds.py` fails if either source changes without this package.

Flag names shared with `quality-evals/speaker_sim/build_run_json.py` mean the same thing there.

How a few of the checks work:
- **Cut-off detection:** with the executor's patch count it is exact (`num_patches >= max_generate_length`). With only a file and a cap, it falls back to the old rule: duration within 10 ms of `max_generate_length × 0.16 s`.
- **Silence layout:** silence is measured in 20 ms frames, and a frame counts as silent when it is more than 40 dB below the loudest frame.
- **Short-circuiting:** a clip that is non-finite, too short or silent isn't measured further.

### Agreement with the old benchmark labels

On the 1,028 archived benchmark WAVs (`scripts/check_against_archive.py`):
- Every measurement, `at_cap` and `signal_ok` value matches `wav_flags.csv`.
- The x-vector runs split exactly as before: the 324 `clean_candidate` clips pass and the 162 cut-off clips fail.
- It also catches 4 near-silent clips (peak about 0.0016) that the old script had no check for. All four were already excluded as continuation-mode output.
