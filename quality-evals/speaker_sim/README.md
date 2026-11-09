# speaker-sim

Speaker similarity (SIM) for the WAT.ai adaptive TTS project: how much a generated clip still sounds like the target voice. It is one of the metrics plotted against NFE alongside WER and latency.

**How it works.** Each clip is converted to 16 kHz mono, turned into a 512-number speaker embedding (a "voiceprint") by a WavLM speaker-verification model, and compared with cosine similarity. 1.0 means an identical voice; lower means less alike.

- Model: [`microsoft/wavlm-base-plus-sv`](https://huggingface.co/microsoft/wavlm-base-plus-sv), pinned to revision `feb593a6c23c1cc3d9510425c29b0a14d2b07b1e` so everyone's numbers come from the same weights.
- SIM values are only comparable when they come from the **same model and revision**. Published TTS papers often use a larger model (WavLM-large, as in seed-tts-eval), so don't compare our numbers to theirs directly.

## Everything lives in this folder

```
speaker-sim/
├── speaker_sim.py        the scorer: two clips in, similarity out
├── sanity_check.py       checks that prove the scorer can be trusted
├── make_test_clips.sh    makes test clips from built-in macOS voices
├── add_recording.sh      converts a real recording into a test clip
├── requirements.txt      exact library versions
├── sanity_results.txt    latest sanity-check output
├── score_clips.py        Phase 2: SIM for every clip in a manifest
├── analyze.py            Phase 2: join WER + latency, averages with 95% CIs, tables, plots
├── plots.py              Phase 2: the plots (used by analyze.py)
├── pipeline_check.py     checks for the Phase 2 code (no model needed, runs in a second)
├── make_dummy_data.py    fake NFE sweep for testing the pipeline
├── prepare_watai_exp.py  turns the team's watai_exp sweeps into pipeline inputs
├── build_run_json.py     one dashboard JSON per run, with invalid clips flagged (see "Per-run JSON")
├── PHASE2_RESULTS.md     what the real sweeps show: SIM, WER, latency vs NFE
├── README.md
├── .venv/                Python environment            (not in git)
├── models/               WavLM weights, ~770 MB        (not in git; downloaded on first run)
├── clips/                16 kHz mono WAV test clips    (not in git; regenerate any time)
├── dummy/                fake sweep + its results       (not in git; make_dummy_data.py)
├── results/watai_exp/    real Phase 2 outputs: tables, plots and WER input in git; intermediates not
│   └── runs/             per-run dashboard JSON + the valid-clips NFE plot
└── recordings/           your original voice memos     (not in git; personal data, never push)
```

The model is stored in `models/` rather than the hidden system cache, so the whole project is self-contained: copy or delete this folder and nothing is left behind elsewhere.

## Setup

```bash
cd quality-evals/speaker_sim   # this folder, inside the team's main repo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Next time, just:

```bash
cd speaker-sim && source .venv/bin/activate
```

The first run downloads the model into `models/`. After that it works offline. A GPU is used if available; a laptop CPU is fine for scoring.

## Usage

```bash
python speaker_sim.py a.wav b.wav        # one pair  -> e.g. 0.9679
python speaker_sim.py --matrix clips/    # every pair of .wav files in a folder
```

From Python:

```python
from speaker_sim import SpeakerSim
sim = SpeakerSim()
score = sim.similarity("generated.wav", "reference.wav")
```

Clips shorter than 0.5 s or (near-)silent raise a `ValueError` instead of returning a misleading score.

## Sanity checks

```bash
bash make_test_clips.sh     # 5 built-in macOS voices x 3 sentences -> clips/
python sanity_check.py      # exits non-zero if any check fails
```

`sanity_check.py` covers every clip in `clips/`: 5 synthetic voices plus 2 real speakers, 21 clips in total. All 16 checks pass (latest output in `sanity_results.txt`).

### Synthetic voices (macOS `say`)

| Check | Result |
|---|---|
| A clip against itself | 1.0000 |
| Same voice, different sentences (average) | 0.966 |
| Different voices (average) | 0.687 |
| Every voice matches itself better than any other voice | yes (5/5) |
| Closest different voices (Moira vs Tessa) | 0.947 |
| AAC round trip (WAV → .m4a → WAV) | 0.980 vs original |
| Silent or < 0.5 s clip | rejected |

### Real voices (Step 6)

Two people, Atharv and Manas, each recorded the same three sentences on Mac Voice Memos (48 kHz AAC, converted to 16 kHz mono WAV). Full matrix in `real_voice_results.txt`.

| Comparison | Mean | Range |
|---|---|---|
| Atharv vs Atharv (different sentences) | **0.969** | 0.961 – 0.976 |
| Manas vs Manas (different sentences) | **0.932** | 0.910 – 0.955 |
| Atharv vs Manas (all 9 pairs) | **0.730** | 0.667 – 0.791 |
| Atharv vs Manas (same sentence only) | 0.732 | 0.672 – 0.791 |

- **Perfect separation:** the lowest same-person score (0.910) is above the highest cross-person score (0.791), a margin of 0.118.
- **It measures the voice, not the words:** two people saying the *same* sentence (0.732) score no higher than saying different sentences (0.730).
- **Format robustness, measured on a real recording (Atharv_1):** 8 kHz 0.985, 44.1 kHz 1.000, stereo 1.000, half volume 0.999 vs original.

### Adding more real voices

Record each of these as its **own** recording, the same sentences the synthetic voices say, so the only thing that differs between clips is the voice:

1. *The quick brown fox jumps over the lazy dog near the river bank.*
2. *Please call Stella and ask her to bring these things from the store.*
3. *Our research team measures speech quality at different step counts.*

Tips: quiet room, normal speaking voice, phone or laptop about 20–30 cm from your mouth, built-in mic (not Bluetooth earbuds), about one second of silence before and after.

Drag the memos from Voice Memos into `recordings/` (macOS blocks reading them from inside the app), name them `Name_n` (n = sentence number), then:

```bash
./add_recording.sh recordings/Name_1.m4a Name_1
python sanity_check.py
```

Clips with the same name prefix are treated as the same speaker.

## What the checks taught us

- **Real speakers separate clearly.** Same person 0.93–0.97, different people 0.73, with no overlap between any pair.
- **Similar voices can score close.** Two different synthetic voices (Moira and Tessa) score 0.947 against each other, nearly as high as each against itself. A SIM difference of about 0.01 between NFE settings may be noise: always show confidence intervals and compare the same prompt and seed across NFE values.
- **Audio format matters a little.** 8 kHz audio and AAC compression each cost about 0.015–0.04 of similarity. Score original WAVs at a consistent sample rate.
- **Recording level barely matters.** Manas's recordings were about 6 dB quieter than Atharv's and still matched each other at 0.93.
- **Reference choice matters.** For the NFE study, compute SIM against (a) the reference voice clip if dots.tts clones voices, and (b) the same prompt and seed at the highest NFE, which isolates the effect of lowering NFE.

## Phase 2: SIM, WER and latency vs NFE

Three steps, one command each:

```bash
python score_clips.py manifest.csv -o results/scored.csv                      # SIM (needs the model)
python analyze.py results/scored.csv --wer wer.csv --latency latency.csv -o results   # everything else
python pipeline_check.py                                                      # checks the logic (62 checks)
```

**1. The manifest** lists every generated clip, one row each:

| column | meaning |
|---|---|
| `run_id` | which experiment/config, e.g. `baseline`, `sweep` |
| `prompt_id` | which text prompt |
| `category` | `short` / `medium` / `long` (blank is fine) |
| `nfe` | NFE used for this clip |
| `seed` | generation seed (blank if deterministic) |
| `wav_path` | the clip (relative to the manifest's folder) |
| `ref_path` | the reference voice clip, if dots.tts clones one (blank otherwise) |

**2. `score_clips.py`** adds, per clip:
- `sim_ref`: clip vs the reference voice.
- `sim_hi_nfe`: clip vs the *same run, prompt and seed* at the highest NFE. This isolates what lowering NFE does. It's 1.0 for the top-NFE clip itself and left blank for a fixed-NFE run (nothing to compare against).
- Clips that can't be scored (too short, silent, unreadable) get an `error` instead of stopping the run.

**3. `analyze.py`** joins WER and latency onto each clip. Each file is matched by whichever of `run_id, prompt_id, nfe, seed` it has (so a latency file with one number per NFE works), or else by audio filename. It stops on duplicates and on clips with no match (`--allow-missing` leaves them blank instead). Column names can be changed with `--wer-col`, `--latency-col`, `--rtf-col`. If there's no RTF column, RTF = latency / audio length.

Outputs:

| file | what's in it |
|---|---|
| `clips.csv` | one row per clip: `run_id, prompt_id, category, nfe, seed, sim_ref, sim_hi_nfe, wer, latency_ms, rtf, ...` |
| `summary_by_nfe.csv` | mean + 95% CI per run, category (and `all`), NFE and metric |
| `paired_deltas.csv` | how much each metric changes vs the same prompt+seed at the highest NFE, mean + 95% CI, and whether the CI excludes 0 |
| `latency_fit.csv` | latency = fixed overhead + cost per NFE × NFE, with 95% CIs |
| `summary.json` | all of the above, for the frontend (blanks are `null`) |
| `plots/` | per run: NFE vs SIM, NFE vs WER, NFE vs latency, latency vs quality, per-prompt heatmap; plus `compare_runs/` |

**How the statistics work:**
- **95% CIs are bootstrap intervals that resample prompts, not clips.** Several seeds of the same prompt aren't independent, so treating them as separate samples would make the intervals look narrower than they really are.
- **Paired deltas are the most sensitive comparison.** Each clip is compared with *its own* prompt and seed at the highest NFE, which cancels out "this prompt is just harder." Phase 1 showed SIM differences of ~0.01 can be noise; a paired delta whose CI excludes 0 is a real change.
- **The latency fit's intercept is the fixed overhead.** The pooled fit can have a low R² because the cost per NFE grows with the length of the audio; the per-category fits (also in `latency_fit.csv`) are tighter.

**Real data: the team's `watai_exp` repo** ([Adaptive-Compute-For-TTS/watai_exp](https://github.com/Adaptive-Compute-For-TTS/watai_exp)). Nick's two full sweeps (54 prompts × NFE 4–32, seed 42, all cloning `experiments/my_files/abhi1.wav`): the original RTX 3090 run and the x-vector RTX 4090 rerun. The sweep data stays in that repo; this folder now lives at `quality-evals/speaker_sim` in the main repo, so clone `watai_exp` next to it and pass its path (`WATAI_EXP` below). Run these from here:

```bash
WATAI_EXP=../../../watai_exp   # path to your clone of the watai_exp repo
python prepare_watai_exp.py $WATAI_EXP -o results/watai_exp
python score_clips.py results/watai_exp/manifest.csv -o results/watai_exp/scored.csv
python analyze.py results/watai_exp/scored.csv --latency results/watai_exp/latency.csv \
    --wer results/watai_exp/wer_arush.csv -o results/watai_exp
python analyze.py results/watai_exp/scored.csv --latency results/watai_exp/latency.csv \
    --wer results/watai_exp/wer_arush.csv --category-col content --no-plots \
    -o results/watai_exp/by_content   # plain / numbers / names / acronyms / prosody / phonetic
```

`wer_arush.csv` is Arush's per-clip ASR output (972 rows, transcripts included). Besides `wer`, the pipeline uses its `cer`, `wer_heard` (WER on the words spoken before a clip was cut off) and `missing_end_words` (→ `truncated`).

**Testing without real data:** `python make_dummy_data.py` builds a fake sweep in `dummy/` (macOS `say` voices with noise and muffling that get stronger at lower "NFE", plus made-up WER and latency). Plots from it are labelled *SYNTHETIC TEST DATA*. In that test, the fit recovered the built-in 180 ms overhead (95% CI 178–183 ms).

## Scoring one run (the worker's Quality Evaluation step)

`evaluate_run.py` is the per-run counterpart of the sweep tools below. The GPU worker calls it with a clip that already passed WAV validation and stores the returned dict as the run's `result["quality"]`:

```python
from evaluate_run import QualityEvaluator

qe = QualityEvaluator(reference_wav="abhi1.wav")    # build once; models load on first use
quality = qe.evaluate(audio_path, prompt, validation=v)   # never raises
# {"scored": True, "sim_ref": 0.93, "wer": 0.0, "cer": 0.0, "transcript": "...",
#  "sim_model": "microsoft/wavlm-base-plus-sv@feb593a...", "asr_model": "faster-whisper-small", "errors": []}
```

- WER/CER and the transcript come from the shared scorer in `quality-evals/wer_cer` (the one behind `wer_arush.csv`), so per-run and sweep numbers match. In the worker's venv, install `requirements-worker.txt` with gpu-executor's constraints (its header has the command). For a standalone setup, install `quality-evals/wer_cer/requirements.txt` as well as the SIM requirements.
- Sai's `result` layout: `validation` (WAV checks), `metrics` (runtime), `quality` (this dict). Quality scoring finishes before the run is marked completed.
- A clip that failed validation is not scored. A scorer that fails leaves its metric `None` and adds an entry to `errors`; the other metric still comes back.
- No `sim_hi_nfe` (needs a second generation at the highest NFE) and no `latency_outlier` (needs the whole run): those stay batch-only.
- Tests: `pytest test_evaluate_run.py` (fakes, no models or torch, but needs `quality-evals/wer_cer/requirements.txt`).

## Per-run JSON for the dashboard

`build_run_json.py` joins SIM (`scored.csv`), WER/CER (Arush's ASR file) and runtime (`latency.csv`) for every clip, **flags the invalid ones**, and writes one file per run:

```bash
python build_run_json.py results/watai_exp/scored.csv --wer results/watai_exp/wer_arush.csv \
    --latency results/watai_exp/latency.csv --flags $WATAI_EXP/experiments/clean_samples/wav_flags.csv \
    -o results/watai_exp/runs --plot
```

`--flags` is the team's `experiments/clean_samples/wav_flags.csv` (use `--no-team-flags` to skip it; contaminated clips then go undetected). Output: `runs/<run_id>.json`, `runs/index.json` (what runs exist and how many clips are valid) and, with `--plot`, `runs/<run_id>_nfe_quality.png`.

**What's in a run file**

| key | contents |
|---|---|
| `config` | seed, model, conditioning, precision, ODE method, guidance, length cap, reference clip and GPU. Read from the run folders' `run_metadata.json`; the GPU label is from the run owner |
| `nfe_definition`, `metrics` | what NFE counts, which SIM model/revision and ASR produced the numbers |
| `validity` | clip counts: total, valid, invalid by flag, cautions by flag, valid clips per NFE and per length bucket |
| `flag_definitions` | every flag with its `severity` (`invalid` or `caution`) and a plain-English description |
| `clips[]` | the frontend's `ExperimentClip` fields (`sim_ref`, `sim_hi_nfe`, `wer`, `cer`, `latency_ms`, `rtf`, ...) plus `content` (prompt type), `valid` and `flags` |
| `summary_by_nfe`, `paired_deltas`, `latency_fit` | the same rows as `summary.json`, but **valid clips only** (`aggregates_cover` says so) |

`clips` is still accepted by the frontend's `parseExperimentClips`, and every field of every clip equals the one in `summary.json`. `summary.json` itself (all clips, invalid ones included) is unchanged.

**Flags.** An *invalid* clip is kept in `clips` with `valid: false` and left out of the aggregates. A *caution* is only noted, and only on clips that are otherwise valid.

| flag | severity | when | found by |
|---|---|---|---|
| `missing_audio` / `unreadable_audio` | invalid | WAV not on disk / can't be decoded | this script |
| `too_short` | invalid | under 0.5 s | this script, the scorer's own rule |
| `silent` | invalid | peak below 0.01 or RMS below 1e-4 | this script, the scorer's own rule |
| `asr_rejected` | invalid | Arush's ASR rejected the clip | `wer_arush.csv` |
| `score_error` | invalid | the SIM scorer failed for any other reason | `scored.csv` |
| `contaminated_continuation` | invalid | reference transcript leaked into the output | team's `wav_flags.csv` |
| `truncated_at_cap` | invalid | stopped at the generation length cap | team's flags, and re-derived from the clip length |
| `missing_metric` | invalid | SIM, WER, CER, latency or RTF is blank | this script |
| `latency_outlier` | caution | RTF over 3× the median of its run, NFE and length bucket | this script |
| `signal_check_failed` | caution | clipping, leading silence, a pause over 1 s or an abrupt end | team's flags |

The silence and length thresholds are read from `speaker_sim.py`'s own source, so the flags can't drift from what the scorer rejects.

**On the real sweeps** (cross-checked against sources that were built independently):

| run | clips | valid | invalid | cautions |
|---|---|---|---|---|
| `xvector_4090` | 486 | **324** | 162 `truncated_at_cap` (all long prompts) | 42 `signal_check_failed` |
| `orig_3090` | 486 | **0** | 486 `contaminated_continuation`, of which 4 `silent` (and the same 4 `asr_rejected`) | - |

- The 324 valid x-vector clips are exactly the team's `clean_candidate` set. My own cap check found the same 162 truncated clips as theirs.
- Three detectors agree on the 4 silent clips: this script's audio check, Arush's ASR and the SIM scorer.
- `orig_3090` has no valid clips, so its file has the flagged clips but empty aggregates. Its all-clips numbers remain in `summary.json`.
- Because the long prompts are all invalid, aggregates cover short and medium prompts only (36 prompts per NFE). Rerun on a sweep with a raised `max_generate_length` (the benchmark script now defaults to 500) to bring long prompts back.

## Next

1. Phase 2 is done: SIM, WER and latency for both `watai_exp` sweeps, summarized in `PHASE2_RESULTS.md`.
2. Phase 3: share the summary with the team, hand `results/watai_exp/runs/` (and `summary.json`) to Shivani's frontend, and confirm the ASR model/normalization with Arush.
3. Worth rerunning once the length cap is raised (and the original run's leaking reference is understood), to see NFE effects on long prompts without truncation.
