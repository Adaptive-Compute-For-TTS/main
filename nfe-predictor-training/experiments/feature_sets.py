"""Candidate text representations for the feature ablation.

A spec string names one representation:

    text5          Shivani's five basic features (the current text5-plus-patch-index-v1 schema)
    rich           extra handcrafted pre-generation counts (RICH_NAMES)
    a+b            concatenation, e.g. text5+rich
    pcaK(spec)     PCA to K components, e.g. pca4(text5+rich)

Every representation is fitted on training texts only and then frozen, so nothing about
validation or test texts leaks into it. The harness appends patch_index afterwards, as
training.dataset.make_patch_vector does. Embedding models register here later as new
base names (features/embeddings.py) and compose with + and pcaK(...) unchanged.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import re
import string

import numpy as np

from features import FEATURE_NAMES, extract_features
from features.embeddings import StaticEmbeddings


@dataclass(frozen=True)
class FittedFeatures:
    """A frozen text -> vector map, plus what it would cost to ship."""

    spec: str
    names: tuple[str, ...]
    transform: Callable[[str], np.ndarray]
    # Numbers the inference bundle would need besides the final mean/std
    # (PCA components, centring). Excludes any external model weights.
    shipped_params: int

    def __call__(self, text: str) -> np.ndarray:
        x = np.asarray(self.transform(text), dtype=np.float64)
        if x.shape != (len(self.names),) or not np.isfinite(x).all():
            raise ValueError(f"{self.spec} returned shape {x.shape} or non-finite values")
        return x

    def matrix(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self.names)))
        return np.stack([self(t) for t in texts])


Fitter = Callable[[Sequence[str]], FittedFeatures]


# --- base representations ------------------------------------------------------------------

def _fit_text5(train_texts: Sequence[str]) -> FittedFeatures:
    del train_texts  # nothing to fit

    def transform(text: str) -> np.ndarray:
        values = extract_features(text)
        return np.array([values[name] for name in FEATURE_NAMES])

    return FittedFeatures("text5", tuple(FEATURE_NAMES), transform, 0)


RICH_NAMES = (
    "sentence_count",                # sentence-final . ! ? groups
    "comma_count",
    "question_exclaim_count",        # ? and !
    "ellipsis_dash_count",           # ... … — – --
    "number_token_count",            # tokens containing a digit
    "acronym_token_count",           # >= 2 capitals and no lower case, e.g. NASA, U.S.
    "capitalized_inner_token_count", # capitalised mid-sentence: a names proxy
    "long_word_count",               # >= 10 letters: a rare-word proxy
    "symbol_count",                  # $ % & @ # / + = * < > ~ ^ | \
    "non_ascii_count",
)
_SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)")
_ELLIPSIS_DASH = re.compile(r"\.\.\.|…|—|–|--")
_SYMBOLS = frozenset("$%&@#/+=*<>~^|\\")


def rich_features(text: str) -> tuple[float, ...]:
    """Deterministic, pre-generation counts in RICH_NAMES order."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    tokens = text.split()
    stripped = [t.strip(string.punctuation) for t in tokens]
    inner_caps = sum(
        1 for i, word in enumerate(stripped)
        if i > 0 and word[:1].isupper() and not tokens[i - 1].endswith((".", "!", "?"))
    )
    acronyms = sum(
        1 for word in stripped
        if sum(c.isupper() for c in word) >= 2 and not any(c.islower() for c in word)
    )
    return tuple(float(v) for v in (
        len(_SENTENCE_END.findall(text)),
        text.count(","),
        sum(c in "?!" for c in text),
        len(_ELLIPSIS_DASH.findall(text)),
        sum(any(c.isdigit() for c in t) for t in tokens),
        acronyms,
        inner_caps,
        sum(word.isalpha() and len(word) >= 10 for word in stripped),
        sum(c in _SYMBOLS for c in text),
        sum(ord(c) > 127 for c in text),
    ))


def _fit_rich(train_texts: Sequence[str]) -> FittedFeatures:
    del train_texts
    return FittedFeatures("rich", RICH_NAMES, lambda t: np.array(rich_features(t)), 0)


BASE_FITTERS: Mapping[str, Fitter] = {"text5": _fit_text5, "rich": _fit_rich}


def static_embedding_fitter(model: StaticEmbeddings) -> Fitter:
    """A base fitter for a loaded static embedding model (nothing to fit).

    shipped_params counts the whole token table: what serving would have to load.
    """
    def fit(train_texts: Sequence[str]) -> FittedFeatures:
        del train_texts
        return FittedFeatures(model.name, tuple(f"{model.name}:{i}" for i in range(model.dim)),
                              model.embed, model.table.size)
    return fit


def with_static_embeddings(models: Mapping[str, StaticEmbeddings],
                           base: Mapping[str, Fitter] = BASE_FITTERS) -> dict[str, Fitter]:
    """A new base-fitter mapping with each model available under its key."""
    clashes = set(models) & set(base)
    if clashes:
        raise ValueError(f"embedding names clash with built-in feature sets: {sorted(clashes)}")
    return {**base, **{name: static_embedding_fitter(m) for name, m in models.items()}}


# --- combinators ---------------------------------------------------------------------------

def _concat(fitters: Sequence[Fitter]) -> Fitter:
    def fit(train_texts: Sequence[str]) -> FittedFeatures:
        parts = [f(train_texts) for f in fitters]
        names = tuple(n for p in parts for n in p.names)
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate feature names in {'+'.join(p.spec for p in parts)}")
        return FittedFeatures(
            "+".join(p.spec for p in parts), names,
            lambda t: np.concatenate([p(t) for p in parts]),
            sum(p.shipped_params for p in parts),
        )
    return fit


def _pca(base: Fitter, k: int) -> Fitter:
    def fit(train_texts: Sequence[str]) -> FittedFeatures:
        inner = base(train_texts)
        x = inner.matrix(train_texts)
        if k < 1 or k > min(x.shape):
            raise ValueError(f"pca{k} needs at least {k} training texts and {k} base features, "
                             f"got {x.shape}")
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        _, _, vt = np.linalg.svd((x - mean) / std, full_matrices=False)
        components = vt[:k]
        # Fix the SVD sign ambiguity so results are reproducible.
        pivots = components[np.arange(k), np.abs(components).argmax(axis=1)]
        components = components * np.where(pivots < 0, -1.0, 1.0)[:, None]
        spec = f"pca{k}({inner.spec})"
        return FittedFeatures(
            spec, tuple(f"{spec}:pc{i}" for i in range(k)),
            lambda t: ((inner(t) - mean) / std) @ components.T,
            inner.shipped_params + components.size + mean.size + std.size,
        )
    return fit


def _split_top_level(spec: str, sep: str) -> list[str]:
    parts, depth, start = [], 0, 0
    for i, c in enumerate(spec):
        depth += (c == "(") - (c == ")")
        if depth < 0:
            raise ValueError(f"unbalanced parentheses in {spec!r}")
        if c == sep and depth == 0:
            parts.append(spec[start:i])
            start = i + 1
    if depth:
        raise ValueError(f"unbalanced parentheses in {spec!r}")
    return [*parts, spec[start:]]


_PCA = re.compile(r"^pca(\d+)\((.+)\)$")


def build(spec: str, base_fitters: Mapping[str, Fitter] = BASE_FITTERS) -> Fitter:
    """Parse a spec string into a fitter. Raises ValueError for unknown names."""
    spec = spec.strip()
    parts = _split_top_level(spec, "+")
    if len(parts) > 1:
        return _concat([build(p, base_fitters) for p in parts])
    match = _PCA.match(spec)
    if match:
        return _pca(build(match[2], base_fitters), int(match[1]))
    if spec in base_fitters:
        return base_fitters[spec]
    raise ValueError(f"unknown feature set {spec!r}; base names: {sorted(base_fitters)}")
