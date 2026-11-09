# wer_cer

Intelligibility scoring for dots.tts clips. Whisper transcribes each clip, and the transcript is compared with the
text the clip was supposed to say. It reports **WER** and **CER** per clip and as corpus totals, plus measures of
cut-off endings. This code produced `wer_arush.csv`, which `../speaker_sim/build_run_json.py` reads.

## Files

| File | What it does |
|---|---|
| `asr.py` | Loads audio (16 kHz mono), rejects clips shorter than 0.5 s or with RMS below 1e-3, transcribes with faster-whisper (`ASR_SETTINGS`: `small`, int8, CPU, English, beam 5, temperature 0) |
| `metrics.py` | `normalize()` (the one normalizer), `compute_wer()`, `wer_cer()`, `corpus_wer()` |
| `wer_scoring.py` | `score_clip()`: per-clip WER/CER two ways (`full` and `no_intro`), cut-off measures, corpus CER. Also a CLI for a `clips.csv` batch |
| `score_demo.py` | Scores one WAV or a manifest into one JSON keyed `run_id__prompt_id__nfe` (official version `no_intro`, uncapped) |
| `tests/` | Unit tests for every metric and normalization rule. They need no audio and no model download |
| `results/` | Saved WER/CER results (see below) |
| `prompts/benchmark_prompts_v1.json` | The 54 benchmark prompts (p001–p054). The tests check normalization against them |

## Results

Saved in `results/`. Corpus numbers are total errors / total reference words; no audio is included.

| Folder | What was scored | ASR | Corpus WER | Corpus CER |
|---|---|---|---|---|
| `results/nfe_benchmark/` | 972 clips: 2 runs (`orig_3090`, `xvector_4090`) × 54 prompts × 9 NFE values | Whisper `small` | 0.583 (`orig_3090` 0.966, `xvector_4090` 0.200) | 0.568 |
| `results/demo_clean_set/` | Demo clean set: 12 prompts × NFE 4/20/32 = 36 clips from `xvector_4090` | `large-v3` / `small` | 0.000 / 0.019 | 0.000 / 0.011 |

- `nfe_benchmark/clips_arush.csv`: per-clip WER/CER, transcripts and error counts. `wer_run_info.json`: ASR
  settings, package versions and normalization rules. `summary.md`: breakdowns by run, NFE and prompt length.
  `intro_and_cut_findings.md`: why `orig_3090` scores so badly (a leaked spoken intro and a 9.12 s length cap).
- `demo_clean_set/wer_cer_demo_<model>.json`: one entry per clip keyed `run_id__prompt_id__nfe`, plus corpus
  totals and totals by NFE. Use the `large-v3` file for the dashboard. All 8 `small` errors are ASR mistakes on
  rare names in p023, not TTS errors.
- This CSV is a newer version of `../speaker_sim/results/watai_exp/wer_arush.csv`: the 2026-10-19 digit-group
  normalization fix changed 2 per-clip WER values and 5 CER values (corpus WER 0.5832 → 0.5831).

## Install and test

```bash
cd quality-evals/wer_cer
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

The Whisper model downloads on the first real transcription, not during the tests.

## Usage

```bash
# One WAV, e.g. right after a generation
python score_demo.py --wav out.wav --text "The kettle is already on the stove." \
    --run-id run_123 --prompt-id p001 --nfe 20 --out wer_cer.json

# Many WAVs: a manifest CSV with audio_path, text, prompt_id, nfe (and optionally run_id)
python score_demo.py --manifest manifest.csv --run-id my_run --out wer_cer.json

# Another Whisper model: same settings otherwise, and always a separate --out
python score_demo.py --manifest manifest.csv --run-id my_run --asr-model large-v3 --out wer_cer_large-v3.json

# Debug a normalization mismatch on typed text
python metrics.py "Your total comes to \$42.75." "your total comes to forty two dollars and seventy five cents"
```

Transcripts are cached in `transcripts_<model>.jsonl` next to the output, keyed by the WAV's SHA-256, so rescoring
never transcribes the same audio twice.

## Calling it from other code

```python
import sys
sys.path.insert(0, "<repo>/quality-evals/wer_cer")
from asr import transcribe                          # same Whisper settings as wer_arush.csv
from wer_scoring import official_columns, score_clip

s = official_columns(score_clip(prompt_text, transcribe(wav_path)), "no_intro")
s["wer"], s["cer"]                                  # capped at 1 and rounded, as in wer_arush.csv
s["wer_uncapped"], s["cer_uncapped"]
```

`transcribe()` raises `asr.AudioRejected` for clips that are too short or silent.

## Scoring rules

The rules are in `metrics.py` and `wer_scoring.py`, and each one has a test.

- Reference and transcript are normalized the same way: lowercase, then `PRE_RULES`, then Whisper's
  `EnglishTextNormalizer` (expands contractions, writes numbers and currency one way), then `POST_RULES`.
- A rule may only reconcile different ways of writing the same spoken words ("Ave." vs "Avenue"). It must never
  turn a wrong word into a right one.
- `no_intro` (the official version) sets aside transcript words that match a re-spoken voice-prompt intro.
  When there is no intro, it gives the same result as `full`.
- Corpus numbers are total errors divided by total reference words. They are never a mean of per-clip values and
  never capped.
- Silent or rejected clips stay in the output with WER = CER = 1.0.
- WER from different ASR models is not comparable. Every output records its model.
