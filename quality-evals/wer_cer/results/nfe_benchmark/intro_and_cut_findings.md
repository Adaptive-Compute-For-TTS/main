# Intro and 9.12 s cut: findings

Sources: our transcripts and scores (`clips_arush.csv` in this folder), a blobless clone of watai_exp (code, metadata,
results.csv; no generated audio), and `experiments/my_files/abhi1.wav` from that repo. Each claim is marked
**CONFIRMED** (checked directly) or **UNVERIFIED** / **LIKELY** (inferred, not checked).

## 1. What the intro is
- **CONFIRMED.** orig_3090 clips start with "Hey, Adam here. Let's create something that feels real, sounds human,
  and connects every time." It is found in 481 of 486 orig clips. Four of the other five are the silent clips; the
  fifth, p035 NFE 16, says "Hello, my name is Adam here." instead; it has no
  exact run of intro words that the alignment prefers over the reference, so no intro is detected (alignment edge
  case, scoring left as is).
- **CONFIRMED.** No xvector_4090 clip has it: `intro_text` is empty on all 486.
- **CONFIRMED.** It is the transcript the dots.tts README pairs with its sample voice
  `docs/_static/audio/male-voice.wav` (watai_exp README.md:487-505). That file is not in studio-dots-ai/dots.tts
  or the Hugging Face model repos, so it was not downloaded.
- **CONFIRMED.** Both runs' `prompts.csv` snapshots use reference_audio `experiments/my_files/abhi1.wav` with that
  sentence as reference_text.
- **CONFIRMED.** The intro audio is generated each time, not pasted in: waveforms of different clips match only for
  the first 0.0001-0.47 s.
- **CONFIRMED.** In 5 orig clips the intro also appears later in the clip (p013 NFE 16, p037 NFE 6 and 16, p040 NFE 4,
  p048 NFE 6), e.g. "did you really drive all hey adam here let us create...". Only a leading intro is set aside,
  so these count as errors.

## 2. Why the orig run has it
- **CONFIRMED.** orig_3090 was generated 2026-10-13 05:11 UTC with `experiments/nfe_benchmark.py` at commit
  91a240b, which always passes `prompt_text=reference_text` (continuation cloning). `--voice-conditioning`
  (default x-vector, no transcript) was added in 65d27f5, 2026-10-13 15:31 EDT, and xvector_4090 used it
  (its run_metadata.json records `voice_conditioning: x-vector`; the orig metadata has no such field).
- **CONFIRMED.** abhi1.wav is 3.69 s (48 kHz mono, 354 KB). With our ASR settings it says:
  **"Hello, my name's Obby and this is a voice test."** It does not say the Adam sentence, so the
  reference_text does not match the reference audio.
- **CONFIRMED.** `src/dots_tts/runtime.py:560` builds the text to speak as `f"{prompt_text}{text}"`: the model is
  asked to say the reference transcript followed by the prompt, with the reference audio prefilled as if it had
  already said the transcript.
- **LIKELY.** Because the prefilled audio ("Hello, my name's Obby...") does not contain the transcript, the model
  speaks the transcript itself, which produces the intro. Supporting evidence: p035 NFE 16 says "Hello, my name is
  Adam here." and p035 NFE 24 "hello run hello man hello...", mixing what abhi1.wav actually says with the
  transcript. Not tested by regenerating audio.

## 3. Where the 9.12 s cap comes from
Hypothesis: 9.12 s = 57 patches = 80 (`--max-generate-length`) minus the reference audio (~23 patches = 3.68 s).
- **CONFIRMED: the reference audio counts toward the limit in continuation mode.** `runtime.py:562` builds a
  single generation schedule with `max_audio_tokens = max_generate_length` (80) audio slots. `model.py`
  (`_prefill`, `_locate_prefill_boundary`) fills the first slots of that same schedule with the reference
  audio's patches, and only the remaining slots are decoded. The first decoded patch is discarded
  (`should_drop_regenerated_prompt_patch`) and only the rest is output.
- **CONFIRMED: 0.16 s per patch** is consistent with the data: xvector (no prefill) is capped at exactly
  12.80 s = 80 x 0.16 s (all 162 long xvector clips), and orig at 9.12 s = 57 x 0.16 s. The value is inferred
  from durations; `patch_size`/`hop_size` are in the model checkpoint config, which is not in the repo.
- **UNVERIFIED: the exact patch count.** abhi1.wav is 23.07 patches long; the code pads it to 24, drops the last
  patch (`prompt_latents_sampled[:, :-patch_size]`) and discards one regenerated patch, which would leave
  80 - 23 - 1 = 56 patches = 8.96 s, one patch short of the observed 57. The difference likely comes from how the
  vocoder frames the reference latents; not checked.
- **CONFIRMED.** 445 of 486 orig clips are exactly 9.12 s; 41 are shorter (7.52-8.96 s), i.e. they ended
  before the cap, and 36 of those finished the sentence. All 972 durations match `audio_duration_seconds` in
  watai_exp results.csv exactly. (The README says every orig clip is exactly 9.12 s.)

## 4. How much of the 9.12 s the intro uses
- **CONFIRMED (sample).** Whisper word timestamps on 40 random orig clips with the full intro: the intro ends at
  min 5.96 s, median 8.01 s, max 9.10 s, mean 7.72 s (85% of 9.12 s).

## 5. Does WER show the endings cut off?
**CONFIRMED: yes.** With the intro set aside (no_intro):
- orig clips at 9.12 s miss on average 22.1 reference words at the end and 92% are unfinished; the 41 shorter orig
  clips miss 0.12 and 12% are unfinished.
- Paired with xvector (same prompt and NFE), orig misses +1.5 more end words on short prompts (58% vs 2%
  unfinished), +15.7 on medium (99% vs 2%) and +32.0 on long (100% vs 97%; xvector long clips are themselves capped
  at 12.8 s).
- None of the target sentence is heard in 14% / 69% / 87% of short / medium / long orig clips.
- `trailing_dels` agrees (19.9 vs 3.7 per clip). The full-version `missing_end_words` understates the cut (12.2)
  because intro words sometimes match reference words by chance.
- `wer_heard` (WER over the part of the sentence before the cut) is 0.0235 for xvector but not meaningful for
  orig: only 1,043 of 10,899 orig reference words are heard at all, and it is dominated by a few clips that loop or
  repeat the intro (0.255 overall, 0.136 without the 5 clips that repeat the intro).
