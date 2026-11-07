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
- `too_short` and `silent` use the speaker-similarity scorer's thresholds (`quality-evals/speaker_sim/speaker_sim.py`), and `build_run_json.py` applies them the same way, at the file's own sample rate. The scorer itself applies them after resampling to 16 kHz, so a clip sitting right at a threshold could still be refused there.
- The five cautions are the `signal_ok` rules from the benchmark's `flag_wav_runs.py`.
- `tests/test_thresholds.py` fails if either source changes without this package.

Flag names shared with `quality-evals/speaker_sim/build_run_json.py` mean the same thing there.

How a few of the checks work:
- **Cut-off detection:** with the executor's patch count it is exact (`num_patches >= max_generate_length`). Each patch is 0.16 s of audio; in gpu-executor's own results `audio_s` is exactly `num_patches × 0.16` for every row. With only a file and a cap, it falls back to the old rule: duration within 10 ms of `max_generate_length × 0.16 s`.
  - `max_generate_length` must be how many patches the model was allowed to *generate*. In x-vector mode (no `prompt_text`, which the team uses) that is just `rt.max_generate_length` (dots.tts default 500).
  - In continuation mode (`prompt_text` passed), dots.tts counts the reference clip's patches against the same cap, so pass the cap minus those patches. Otherwise cut-offs are missed.
  - dots.tts counts the reference *after* trimming its leading/trailing silence (librosa `top_db=30`) and resampling to 48 kHz: `ceil(trimmed samples / 7680)`. For `abhi1.wav` that is 23 patches; counting the untrimmed file gives 24, which would wrongly flag a natural 56-patch clip.
  - This is why the archived continuation runs stopped at 9.12 s (57 = 80 − 23 patches) while the x-vector runs stopped at 12.8 s.
- **Silence layout:** silence is measured in 20 ms frames, and a frame counts as silent when it is more than 40 dB below the loudest frame.
- **Short-circuiting:** a clip that is non-finite, too short or silent isn't measured further.

### Agreement with the old benchmark labels

On the 1,028 archived benchmark WAVs (`scripts/check_against_archive.py`):
- For the 1,024 clips both scripts measure, every stored measurement, `at_cap` and `signal_ok` value matches `wav_flags.csv`.
- The other 4 are near-silent clips (peak about 0.0016) that the old script had no check for. The new code rejects them as `silent` before measuring silence layout; their duration, peak and `at_cap` still match. All four were already excluded as continuation-mode output.
- The x-vector runs split exactly as before: the 324 `clean_candidate` clips pass and the 162 cut-off clips fail.

## Setup

From the repository root:

```bash
python -m venv wav-validation/.venv          # or install into the GPU worker's env
wav-validation/.venv/bin/pip install -e 'wav-validation[test]'
wav-validation/.venv/bin/pytest wav-validation/tests
```

## Calling it from the GPU worker

```python
from adaptive_nfe import generate_fixed_nfe          # gpu-executor/
from wav_validation import validate_and_store

result = generate_fixed_nfe(rt, nfe=run.nfe, text=run.prompt)
audio = result.output["audio"].float().cpu().squeeze().numpy()   # same conversion as generate.py
v = validate_and_store(
    audio, result.output["sample_rate"], run.id, STORAGE_DIR,
    num_patches=result.num_patches,
    max_generate_length=rt.max_generate_length,   # x-vector mode; see "Cut-off detection" for continuation mode
)
if v["ok"]:
    # status=completed, audio_path=v["audio_path"], result["validation"]=v
    # then hand v["audio_path"] to quality evaluation
    ...
else:
    # status=failed, error_message="WAV failed validation: " + ", ".join(v["flags"])
    ...
```

What it writes under `STORAGE_DIR`:

```
<run_id>/audio.wav                   passed: the run's audio_path (float32 WAV, like generate.py)
<run_id>/validation.json             the result below
_rejected/<run_id>/audio.wav         failed: kept for debugging, never served
_rejected/<run_id>/validation.json
```

Guarantees:
- **Atomic writes:** both files are written via a temp file and a rename, so the results API never sees a half-written file. They get normal file permissions (from the umask), not owner-only.
- **Each run is stored once:** a second call with the same `run_id` raises `FileExistsError` and leaves the first result untouched. This holds even when two workers store the same run at the same moment with different verdicts (e.g. a stuck run retried while the original finishes): every store claims `<run_id>/` with an atomic `mkdir` first, and a failed run releases that claim only after `_rejected/<run_id>/` exists.
- **A failed write can be retried:** if writing fails partway (e.g. disk full), everything that call created is removed before the error is raised, so the run isn't blocked.
- **Safe folder names:** `run_id` must be a plain name (backend UUIDs are), so it can't escape `STORAGE_DIR`.

`validate_audio(audio, sample_rate, ...)` does the same checks without saving anything. `validate_wav(path, ...)` checks a file that's already on disk, and never raises for a bad file.

### Result shape (draft, to confirm with gpu-workers: Tony / Eric)

```json
{
  "ok": true,
  "flags": ["abrupt_end"],
  "signal_ok": false,
  "checks": {
    "duration_s": 9.12, "sample_rate": 48000, "channels": 1,
    "peak": 0.4419, "rms": 0.05, "non_finite_samples": 0, "clipped_samples": 0,
    "leading_silence_s": 0.12, "trailing_silence_s": 0.0, "longest_gap_s": 0.66,
    "num_patches": 57, "max_generate_length": 500, "cap_s": 80.0, "at_cap": false, "cap_source": "patches"
  },
  "thresholds": {"min_seconds": 0.5, "...": "..."},
  "version": "0.1.0",
  "run_id": "…", "audio_path": "/…/<run_id>/audio.wav", "stored_path": "/…/<run_id>/audio.wav"
}
```

What the fields mean:
- `ok` is false only when there's an invalid flag.
- `signal_ok` is true when there are no caution flags, which matches the old `wav_flags.csv` column. It is `null` when the clip wasn't measured.
- `cap_source` is `"patches"`, `"duration"` or `null` (no cap given).
- `run_id`, `audio_path` and `stored_path` only appear in `validate_and_store`'s result.

## Checking files by hand (e.g. on the GPU box)

```bash
python -m wav_validation runs/                                   # every *.wav under runs/
python -m wav_validation audio.wav --max-generate-length 500     # also check for cut-off audio
python -m wav_validation runs/ --json > report.json              # full results; summary on stderr
```

Output is one `OK`/`FAIL` line per file with its flags, then a summary:

```
FAIL  .../output_p027_nfe8.wav  truncated_at_cap, abrupt_end
243 files: 162 ok, 81 failed (truncated_at_cap 81, abrupt_end 67, long_gap 51, leading_silence 21)
```

The exit code is 1 if any file failed. Cautions alone still exit 0.

## The old benchmark files

These are copied unchanged from `nicholaskim17/watai_exp` (`experiments/clean_samples/`, commit `7555334`):

| File | What it is |
|---|---|
| `flag_wav_runs.py` | The original batch script: labels every archived benchmark WAV `clean_candidate`, `truncated_at_cap` or `contaminated_continuation`. It only runs inside the watai_exp repo. |
| `wav_flags.csv` | Its output (1,028 rows). `quality-evals/speaker_sim/build_run_json.py --flags` reads this file, so keep its columns. |
| `run_summary.json` | Per-NFE and per-length-bucket medians for the two full runs. |

`flag_wav_runs.py` still carries its own copy of the thresholds. Rebuilding it on top of this package is a planned follow-up. Until then, this command shows the two still agree:

```bash
wav-validation/.venv/bin/python wav-validation/scripts/check_against_archive.py --archive /path/to/watai_exp
WAV_VALIDATION_ARCHIVE=/path/to/watai_exp wav-validation/.venv/bin/pytest wav-validation/tests   # + real-audio tests
```
