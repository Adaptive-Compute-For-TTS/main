# Phase 1 notes and Phase 2 handoff

Read this together with `README.md` to continue the work in a new chat.

## Context

- Project: WAT.ai adaptive TTS (University of Waterloo). The team is building an adaptive compute scheduler for the **dots.tts** flow-matching TTS model. The long-term goal is to use as few **NFE** (number of function evaluations = model passes per generation) as possible without hurting quality.
- Atharv's task: "Build speaker similarity metrics and plots comparing NFE against metrics such as similarity, WER, and latency."
- Sai (lead) wants the team to first understand how NFE relates to WER, SIM and latency (including how much latency is fixed overhead vs per-NFE cost).
- Teammates whose outputs this depends on:
  - **Nishant**: fixed-NFE baseline WAVs and config.
  - **Nick**: NFE sweep and NFE vs latency.
  - **Eric**: benchmark prompt set with short/medium/long labels.
  - **Arush**: ASR and WER scripts.
  - **Afreed**: latency benchmarking.
  - **Niranjan**: GPU telemetry.
  - **Shivani**: frontend that will show results.
- WATcloud (WATonomous cluster) access has been requested. Only needed if Atharv runs dots.tts himself.

## Phase 1 (done): build and validate the SIM metric

- Scorer: `speaker_sim.py`. Audio is converted to 16 kHz mono, turned into a 512-dim speaker embedding by `microsoft/wavlm-base-plus-sv` (pinned revision `feb593a6c23c1cc3d9510425c29b0a14d2b07b1e`, stored in `models/`, works offline), and compared with cosine similarity.
- Sanity checks (`sanity_check.py`): 16/16 pass on 21 clips (5 macOS `say` voices + Atharv + Manas).
- Key numbers:
  - **Synthetic voices:** same voice 0.966, different 0.687. Closest different pair: Moira–Tessa 0.947.
  - **Real voices:**
    - Atharv–Atharv 0.969
    - Manas–Manas 0.932
    - Atharv–Manas 0.730 (same sentence 0.732)
    - Lowest same-person score 0.910 vs highest cross-person score 0.791, margin 0.118, no overlap
  - **Format robustness (real clip):** 8 kHz 0.985, 44.1 kHz 1.000, stereo 1.000, half volume 0.999. AAC round trip 0.980.
- Lessons:
  - SIM differences of about 0.01–0.02 can be noise: compare the same prompt and seed across NFE values, and report confidence intervals.
  - Score original WAVs at a consistent sample rate.
  - Compute SIM against both the reference voice (if dots.tts clones voices) and the same prompt and seed at the highest NFE.
- Environment: Python 3.14 venv in `.venv` (torch 2.14.1, transformers 5.19.0, librosa 1.0.0, soundfile 0.14.0, numpy 2.5.3), macOS Apple Silicon.
- Git: local commits `05a21d3`, `bf58b36` (plus this notes file). **No remote, nothing pushed.** Ask the team where the code should live before pushing. `models/`, `clips/` and `recordings/` are git-ignored; recordings are personal data and must never be pushed.

## Phase 2 (next): score real dots.tts outputs and plot against NFE

1. Get WAVs, configs, NFE values and seeds from Nishant (baseline) and Nick (sweep), WER code from Arush, and latency data from Afreed.
2. Score SIM per clip:
   - `sim_ref` against the reference voice, if any
   - `sim_hi_nfe` against the same prompt and seed at the highest NFE
3. Build one table, one row per clip: `run_id, prompt_id, category, nfe, seed, sim_ref, sim_hi_nfe, wer, latency_ms, rtf`.
4. Aggregate per NFE (and per short/medium/long category) as mean plus 95% bootstrap CI.
5. Plots (matplotlib):
   - NFE vs SIM
   - NFE vs WER
   - NFE vs latency (expect roughly linear; the intercept is fixed overhead)
   - latency vs quality, labelled by NFE (find the knee)
   - optional: which prompts degrade first
6. Write a short summary of what the curves show, and hand the CSV/JSON to Shivani's frontend.
