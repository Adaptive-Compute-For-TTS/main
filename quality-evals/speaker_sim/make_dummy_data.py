"""Make a fake NFE sweep to test the Phase 2 pipeline before the real dots.tts files arrive.

NOT real TTS output. Each "generated clip" is a macOS `say` voice reading a prompt; lower
"NFE" is imitated by adding noise and muffling (low-pass), stronger at lower NFE and a bit
stronger for long prompts. WER and latency are made up with simple formulas. The point is
only to check that scoring, joining, statistics and plots behave as expected.

Writes dummy/ (git-ignored):
  manifest.csv   clip list for score_clips.py   (run_ids "dummy_sweep" and "dummy_baseline")
  wer.csv        matched by filename            (tests the filename join)
  latency.csv    matched by run/prompt/nfe/seed (tests the key join)
  refs/ clean/ clips/   audio
"""
import csv
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

OUT = Path("dummy")
SR = 16_000
VOICES = ["Daniel", "Karen", "Rishi", "Tessa"]
REF_TEXT = "This is my reference recording, used to check that the generated voice still sounds like me."
PROMPTS = {
    "short": ["Good morning, everyone.", "Please close the door."],
    "medium": ["The library opens at nine and closes late on weekdays.",
               "We tested the new model on a set of sentences of different lengths."],
    "long": ["When the weather is clear, you can see the mountains from the top floor, and on a really good day "
             "the lake shines in the distance like a sheet of glass.",
             "Before you submit the report, please double check every number in the tables, make sure the "
             "figures match the text, and send a copy to the whole team by Friday afternoon."],
}
SWEEP_NFE = [4, 8, 16, 32]
SWEEP_SEEDS = [0, 1]
BASELINE_NFE, BASELINE_SEEDS = 16, [0, 1, 2]


def say(voice: str, text: str, path: Path) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        subprocess.run(["say", "-v", voice, "-o", str(path), "--file-format=WAVE", "--data-format=LEI16@16000", text],
                       check=True)
    wav, sr = sf.read(path, dtype="float32")
    assert sr == SR
    return wav


def degrade(wav: np.ndarray, nfe: int, category: str, rng: np.random.Generator) -> np.ndarray:
    steps = np.log2(nfe / 4)  # 0 at NFE 4, 3 at NFE 32
    snr_db = 6 + 13 * steps - (4 if category == "long" else 0) + rng.normal(0, 1.5)
    out = wav.copy()
    if steps < 2:  # muffle below NFE 16: cutoff 2 kHz at NFE 4, 4 kHz at NFE 8
        out = sosfilt(butter(4, 2000 * 2**steps, fs=SR, output="sos"), out).astype("float32")
    power = np.mean(out**2)
    noise = rng.normal(0, np.sqrt(power / 10 ** (snr_db / 10)), out.shape)
    return (out + noise).astype("float32")


def main() -> None:
    refs = {v: OUT / "refs" / f"{v}_ref.wav" for v in VOICES}
    for v, p in refs.items():
        say(v, REF_TEXT, p)

    (OUT / "clips").mkdir(parents=True, exist_ok=True)
    manifest, wer_rows, lat_rows = [], [], []
    rng = np.random.default_rng(42)
    p_i = 0
    for voice in VOICES:
        for category, texts in PROMPTS.items():
            for t_i, text in enumerate(texts):
                p_i += 1
                prompt_id = f"{voice.lower()}_{category}{t_i + 1}"
                clean = say(voice, text, OUT / "clean" / f"{prompt_id}.wav")
                dur = len(clean) / SR
                runs = [("dummy_sweep", n, s) for n in SWEEP_NFE for s in SWEEP_SEEDS]
                runs += [("dummy_baseline", BASELINE_NFE, s) for s in BASELINE_SEEDS]
                for run_id, nfe, seed in runs:
                    name = f"{run_id}__{prompt_id}__nfe{nfe}__s{seed}.wav"
                    noise_rng = np.random.default_rng([p_i, nfe, seed])  # same audio every time
                    sf.write(OUT / "clips" / name, degrade(clean, nfe, category, noise_rng), SR)
                    manifest.append(dict(run_id=run_id, prompt_id=prompt_id, category=category, nfe=nfe, seed=seed,
                                         wav_path=f"clips/{name}", ref_path=f"refs/{voice}_ref.wav"))
                    wer = 0.03 + (0.02 if category == "short" else 0) + 0.4 * np.exp(-(nfe - 4) / 3.5)
                    wer_rows.append(dict(file=name, wer=round(float(np.clip(wer + rng.normal(0, 0.02), 0, 1)), 4)))
                    lat = 180 + nfe * 9.0 * dur + rng.normal(0, 15)
                    lat_rows.append(dict(run_id=run_id, prompt_id=prompt_id, nfe=nfe, seed=seed,
                                         latency_ms=round(float(lat), 1)))

    for name, rows in [("manifest.csv", manifest), ("wer.csv", wer_rows), ("latency.csv", lat_rows)]:
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"dummy data: {len(manifest)} clips, {len(VOICES)} voices, "
          f"{sum(len(t) for t in PROMPTS.values()) * len(VOICES)} prompts -> {OUT}/")


if __name__ == "__main__":
    main()
