"""Embedders: a protocol plus a deterministic, offline hashing embedder.

``HashingEmbedder`` turns text into a fixed-width numpy vector using hashed
token n-grams weighted with a smoothed IDF-style factor. It needs no model
weights, no downloads and no API keys — which is a *feature*: the whole
service (demo, tests, CI) runs hermetically offline. The ``Embedder`` protocol
is the seam where a real embedding model (OpenAI, sentence-transformers, ...)
plugs in without touching anything else.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    """Anything that can map texts to fixed-width float vectors."""

    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` float32 array, L2-normalised rows."""
        ...


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; the shared tokenizer for BM25 and hashing."""
    return _TOKEN_RE.findall(text.lower())


class HashingEmbedder:
    """Deterministic hashing/TF-IDF-style embedder (the "feature" embedder).

    Each token and each word bigram is hashed into ``dim`` buckets with signed
    hashing (the "hashing trick"), weighted by ``1 + log(tf)`` and a document
    frequency table learned via ``fit`` (a smoothed IDF-style factor; unseen
    terms get weight 1.0). Rows are L2-normalised so dot product == cosine.
    """

    def __init__(self, dim: int = 256, use_bigrams: bool = True) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.use_bigrams = use_bigrams
        self._df: dict[str, int] = {}
        self._n_docs = 0

    # -- corpus statistics (optional; embedder works without fit) ----------

    def fit(self, texts: list[str]) -> HashingEmbedder:
        """Learn document frequencies for an IDF-style weighting."""
        self._df = {}
        self._n_docs = 0
        for t in texts:
            self._n_docs += 1
            for tok in set(tokenize(t)):
                self._df[tok] = self._df.get(tok, 0) + 1
        return self

    def _weight(self, token: str) -> float:
        df = self._df.get(token, 0)
        if self._n_docs == 0 or df == 0:
            return 1.0
        return math.log((self._n_docs + 1) / (df + 1)) + 1.0

    # -- embedding ----------------------------------------------------------

    def _features(self, text: str) -> list[str]:
        toks = tokenize(text)
        feats = list(toks)
        if self.use_bigrams:
            feats.extend(f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False))
        return feats

    def _bucket(self, feat: str) -> tuple[int, float]:
        h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        code = int.from_bytes(h, "big")
        idx = code % self.dim
        sign = 1.0 if (code >> 63) & 1 else -1.0
        return idx, sign

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[str, int] = {}
            for feat in self._features(text):
                counts[feat] = counts.get(feat, 0) + 1
            for feat, tf in counts.items():
                idx, sign = self._bucket(feat)
                out[row, idx] += sign * (1.0 + math.log(tf)) * self._weight(feat)
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
        return out
