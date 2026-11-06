"""Speaker similarity (SIM) between audio clips.

Each clip -> 16 kHz mono -> speaker embedding (WavLM speaker-verification model)
-> cosine similarity. 1.0 = identical voice, lower = less alike.

Usage:
  python speaker_sim.py a.wav b.wav          # one pair
  python speaker_sim.py --matrix clips/       # every pair in a folder
"""
import argparse
import itertools
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, WavLMForXVector

MODEL_ID = "microsoft/wavlm-base-plus-sv"
# Pinned so every teammate's SIM numbers come from the exact same weights.
MODEL_REVISION = "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e"
# Weights live inside this project (downloaded here on first run), not in a hidden system cache.
MODEL_DIR = Path(__file__).resolve().parent / "models"
SR = 16_000
MIN_SECONDS = 0.5
# Loudest sample must reach -40 dBFS. Real speech here peaks at 0.09 or more; failed dots.tts
# generations (near-silent, peak ~0.002) would otherwise get a plausible-looking score.
MIN_PEAK = 0.01


class SpeakerSim:
    def __init__(self, model_id: str = MODEL_ID, revision: str = MODEL_REVISION, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fe = AutoFeatureExtractor.from_pretrained(model_id, revision=revision, cache_dir=MODEL_DIR)
        self.model = WavLMForXVector.from_pretrained(model_id, revision=revision, cache_dir=MODEL_DIR).to(self.device).eval()
        self._cache: dict[str, torch.Tensor] = {}

    @staticmethod
    def load(path: str | Path) -> np.ndarray:
        """Any format, sample rate or channel count -> 16 kHz mono float32."""
        wav, _ = librosa.load(path, sr=SR, mono=True)
        if wav.size < SR * MIN_SECONDS:
            raise ValueError(f"{path}: shorter than {MIN_SECONDS}s")
        if np.sqrt(np.mean(wav**2)) < 1e-4 or np.abs(wav).max() < MIN_PEAK:
            raise ValueError(f"{path}: (near-)silent audio")
        return wav

    @torch.no_grad()
    def embed(self, path: str | Path) -> torch.Tensor:
        key = str(Path(path).resolve())
        if key not in self._cache:  # each file is embedded once
            inputs = self.fe(self.load(path), sampling_rate=SR, return_tensors="pt").to(self.device)
            emb = self.model(**inputs).embeddings[0]
            self._cache[key] = torch.nn.functional.normalize(emb, dim=-1).cpu()
        return self._cache[key]

    def similarity(self, a: str | Path, b: str | Path) -> float:
        return float(self.embed(a) @ self.embed(b))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", help="two audio files to compare")
    p.add_argument("--matrix", metavar="DIR", help="compare every pair of .wav files in DIR")
    args = p.parse_args()
    sim = SpeakerSim()

    if args.matrix:
        files = sorted(Path(args.matrix).glob("*.wav"))
        for a, b in itertools.combinations(files, 2):
            print(f"{sim.similarity(a, b):.4f}  {a.name}  {b.name}")
    elif len(args.files) == 2:
        print(f"{sim.similarity(*args.files):.4f}")
    else:
        p.error("give two files, or --matrix DIR")


if __name__ == "__main__":
    main()
