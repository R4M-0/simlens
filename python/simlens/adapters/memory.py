"""In-memory id -> vector store (zero dependencies)."""
from __future__ import annotations

import numpy as np

from .base import Sample


class Memory:
    """Trivial in-memory vector store with optional per-item payloads."""

    def __init__(self, vectors: dict | None = None):
        self._v: dict = {}
        self._p: dict = {}
        if vectors:
            for k, v in vectors.items():
                self.add(k, v)

    def add(self, id, vector, payload: dict | None = None) -> "Memory":
        self._v[id] = np.asarray(vector, dtype=np.float32).ravel()
        if payload is not None:
            self._p[id] = payload
        return self

    def get(self, ids: list) -> np.ndarray:
        return np.stack([self._v[i] for i in ids])

    def sample(self, n: int) -> Sample:
        ids = list(self._v)[:n]
        vectors = np.stack([self._v[i] for i in ids]) if ids else np.empty((0, 0), np.float32)
        payloads = [self._p.get(i, {}) for i in ids]
        return Sample(vectors=vectors, payloads=payloads, ids=ids)

    def __len__(self) -> int:
        return len(self._v)
