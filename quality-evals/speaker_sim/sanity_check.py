"""Sanity checks for speaker_sim.py. Run after make_test_clips.sh.

Clips are named <Voice>_<n>.wav: same voice, different sentences.
"""
import itertools
import statistics
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from speaker_sim import SR, SpeakerSim

CLIPS = Path("clips")
sim = SpeakerSim()
failures = 0


def check(name: str, ok: bool, detail: str) -> None:
    global failures
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")


files = sorted(CLIPS.glob("*.wav"))
voice = lambda f: f.stem.rsplit("_", 1)[0]

# 1. A clip against itself is ~1.0
a = files[0]
s = sim.similarity(a, a)
check("identical clip", s > 0.999, f"{s:.4f}")

# 2. Each voice matches itself better than it matches any other voice
pairs: dict[tuple[str, str], list[float]] = {}
for x, y in itertools.combinations(files, 2):
    pairs.setdefault(tuple(sorted((voice(x), voice(y)))), []).append(sim.similarity(x, y))
mean = {k: statistics.mean(v) for k, v in pairs.items()}
same = [s for k, v in pairs.items() if k[0] == k[1] for s in v]
diff = [s for k, v in pairs.items() if k[0] != k[1] for s in v]
voices = sorted({voice(f) for f in files})
for v in voices:
    own = mean[(v, v)]
    nearest, score = max(((o, mean[tuple(sorted((v, o)))]) for o in voices if o != v), key=lambda t: t[1])
    check(f"{v} matches itself best", own > score, f"self {own:.3f} vs closest other ({nearest}) {score:.3f}")
gap = statistics.mean(same) - statistics.mean(diff)
check("average gap", gap > 0.15, f"same-voice {statistics.mean(same):.3f} vs different {statistics.mean(diff):.3f} (gap {gap:.3f})")
closest = max((k for k in mean if k[0] != k[1]), key=mean.get)
print(f"note  most similar different voices: {closest[0]} & {closest[1]} at {mean[closest]:.3f}")

# 3. Robust to format: other sample rates, stereo, quieter
wav, _ = librosa.load(a, sr=SR, mono=True)
with tempfile.TemporaryDirectory() as tmp:
    variants = {
        "8 kHz": (librosa.resample(wav, orig_sr=SR, target_sr=8000), 8000),
        "44.1 kHz": (librosa.resample(wav, orig_sr=SR, target_sr=44100), 44100),
        "stereo": (np.stack([wav, wav], axis=1), SR),
        "half volume": (wav * 0.5, SR),
    }
    for name, (data, sr) in variants.items():
        path = Path(tmp) / f"{name}.wav"
        sf.write(path, data, sr)
        s = sim.similarity(a, path)
        check(f"format: {name}", s > 0.95, f"{s:.4f} vs original")

    # 4. Bad input fails loudly instead of returning a number
    for name, data in {"silence": np.zeros(SR * 2), "too short": wav[: SR // 10]}.items():
        path = Path(tmp) / f"{name}.wav"
        sf.write(path, data, SR)
        try:
            sim.similarity(a, path)
            check(f"rejects {name}", False, "returned a score")
        except ValueError as e:
            check(f"rejects {name}", True, str(e).split(": ", 1)[1])

print(f"\n{len(files)} clips, {len(same)} same-voice pairs, {len(diff)} different-voice pairs")
sys.exit(1 if failures else 0)
