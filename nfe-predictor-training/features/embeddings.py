"""Embedding features for NFE prediction: static (model2vec-style) token embeddings.

A static model is one vector per vocabulary token. A text's embedding is the mean of
its tokens' vectors, then optionally L2-normalized. There is no transformer
forward pass, so this needs numpy only: the WordPiece tokenizer below
re-implements Hugging Face's BertNormalizer + BertPreTokenizer + WordPiece, and the
weights are read straight from model.safetensors. Inference therefore takes no
torch, `tokenizers` or `model2vec` dependency.

Tested against minishlab/potion-base-8M (256 dims, 29,528 tokens, ~29 MB). Download
it outside the repo, e.g. `hf download minishlab/potion-base-8M --local-dir DIR`.
Weights are never committed.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct
import unicodedata

import numpy as np

_SAFETENSORS_DTYPES = {"F32": np.float32, "F16": np.float16, "F64": np.float64}


def read_safetensors(path: str | Path) -> dict[str, np.ndarray]:
    """Read every tensor from a .safetensors file (8-byte header size, JSON header, raw data)."""
    data = Path(path).read_bytes()
    (header_size,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8:8 + header_size])
    body = memoryview(data)[8 + header_size:]
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if info["dtype"] not in _SAFETENSORS_DTYPES:
            raise ValueError(f"unsupported safetensors dtype {info['dtype']} for {name}")
        start, end = info["data_offsets"]
        array = np.frombuffer(body[start:end], dtype=_SAFETENSORS_DTYPES[info["dtype"]])
        tensors[name] = array.reshape(info["shape"]).astype(np.float64)
    return tensors


# --- BERT WordPiece, matching Hugging Face tokenizers -----------------------------------------

def _is_control(c: str) -> bool:
    return c not in "\t\n\r" and unicodedata.category(c).startswith("C")


def _is_whitespace(c: str) -> bool:
    return c in " \t\n\r" or unicodedata.category(c) == "Zs"


def _is_punctuation(c: str) -> bool:
    o = ord(c)
    ascii_punct = 33 <= o <= 47 or 58 <= o <= 64 or 91 <= o <= 96 or 123 <= o <= 126
    return ascii_punct or unicodedata.category(c).startswith("P")


def _is_cjk(c: str) -> bool:
    o = ord(c)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0x20000 <= o <= 0x2A6DF
            or 0x2A700 <= o <= 0x2B73F or 0x2B740 <= o <= 0x2B81F or 0x2B820 <= o <= 0x2CEAF
            or 0xF900 <= o <= 0xFAFF or 0x2F800 <= o <= 0x2FA1F)


def bert_normalize(text: str, *, lowercase: bool = True, strip_accents: bool | None = None) -> str:
    """BertNormalizer: clean text, pad CJK, strip accents (follows lowercase if None), lowercase."""
    cleaned = "".join(
        " " if _is_whitespace(c) else f" {c} " if _is_cjk(c) else c
        for c in text if c not in "\0�" and not _is_control(c)
    )
    if strip_accents if strip_accents is not None else lowercase:
        cleaned = "".join(c for c in unicodedata.normalize("NFD", cleaned)
                          if unicodedata.category(c) != "Mn")
    return cleaned.lower() if lowercase else cleaned


def bert_pre_tokenize(text: str) -> list[str]:
    """Split on whitespace, and make every punctuation character its own word."""
    words, current = [], []
    for c in text:
        if _is_whitespace(c) or _is_punctuation(c):
            if current:
                words.append("".join(current))
                current = []
            if _is_punctuation(c):
                words.append(c)
        else:
            current.append(c)
    return [*words, "".join(current)] if current else words


@dataclass(frozen=True)
class WordPiece:
    vocab: dict[str, int]
    unk_token: str = "[UNK]"
    prefix: str = "##"
    max_chars: int = 100
    lowercase: bool = True
    strip_accents: bool | None = None

    @classmethod
    def from_tokenizer_json(cls, path: str | Path) -> "WordPiece":
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        model, normalizer = spec["model"], spec.get("normalizer") or {}
        if model.get("type") != "WordPiece" or (spec.get("pre_tokenizer") or {}).get("type") != "BertPreTokenizer":
            raise ValueError("only BERT WordPiece tokenizers are supported")
        if normalizer.get("type") != "BertNormalizer":
            raise ValueError("only BertNormalizer is supported")
        return cls(dict(model["vocab"]), model["unk_token"], model["continuing_subword_prefix"],
                   model["max_input_chars_per_word"], normalizer.get("lowercase", True),
                   normalizer.get("strip_accents"))

    def _word_ids(self, word: str) -> list[int]:
        if len(word) > self.max_chars:
            return [self.vocab[self.unk_token]]
        ids, start = [], 0
        while start < len(word):
            for end in range(len(word), start, -1):
                piece = word[start:end] if start == 0 else self.prefix + word[start:end]
                if piece in self.vocab:
                    ids.append(self.vocab[piece])
                    start = end
                    break
            else:
                return [self.vocab[self.unk_token]]
        return ids

    def encode(self, text: str) -> list[int]:
        """Token ids without special tokens ([CLS]/[SEP])."""
        normalized = bert_normalize(text, lowercase=self.lowercase, strip_accents=self.strip_accents)
        return [i for word in bert_pre_tokenize(normalized) for i in self._word_ids(word)]


# --- static embedding model --------------------------------------------------------------------

@dataclass(frozen=True)
class TokenMeanTable:
    """text -> mean of its known tokens' rows; `empty` when it has none.

    Because the output is a mean (weights sum to 1), any row-wise affine map f
    satisfies f(mean(rows)) == mean(f(rows)). `map_affine` uses this to fold
    standardization and PCA into the table itself, so serving ships [vocab, K].
    """

    tokenizer: WordPiece
    table: np.ndarray  # [vocab, dim]
    empty: np.ndarray  # [dim]

    def __call__(self, text: str) -> np.ndarray:
        unk = self.tokenizer.vocab[self.tokenizer.unk_token]
        ids = [i for i in self.tokenizer.encode(text) if i != unk]
        return self.table[ids].mean(axis=0) if ids else self.empty

    def map_affine(self, fn) -> "TokenMeanTable":
        """A new table with fn (row-wise affine: [n, dim] -> [n, k]) applied to every row."""
        return TokenMeanTable(self.tokenizer, fn(self.table), fn(self.empty[None, :])[0])

    @property
    def size(self) -> int:
        return self.table.size + self.empty.size


@dataclass(frozen=True)
class StaticEmbeddings:
    name: str
    tokenizer: WordPiece
    table: np.ndarray  # [vocab, dim]
    normalize: bool

    @classmethod
    def load(cls, directory: str | Path, name: str | None = None) -> "StaticEmbeddings":
        """Load a model2vec directory: model.safetensors, tokenizer.json, config.json."""
        directory = Path(directory)
        tensors = read_safetensors(directory / "model.safetensors")
        if "embeddings" not in tensors or tensors["embeddings"].ndim != 2:
            raise ValueError(f"{directory} has no 2-D 'embeddings' tensor")
        tokenizer = WordPiece.from_tokenizer_json(directory / "tokenizer.json")
        table = tensors["embeddings"]
        if table.shape[0] < max(tokenizer.vocab.values()) + 1:
            raise ValueError("embedding table is smaller than the tokenizer vocabulary")
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        return cls(name or directory.name, tokenizer, table, bool(config.get("normalize", True)))

    @property
    def dim(self) -> int:
        return self.table.shape[1]

    @property
    def token_mean(self) -> TokenMeanTable:
        return TokenMeanTable(self.tokenizer, self.table, np.zeros(self.dim))

    def embed(self, text: str) -> np.ndarray:
        """Mean of the token vectors ([UNK] dropped, as model2vec does); zeros for no tokens."""
        vector = self.token_mean(text)
        if self.normalize:
            norm = np.linalg.norm(vector)
            vector = vector / norm if norm > 0 else vector
        return vector
