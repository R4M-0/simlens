"""Vector-store adapters: fetch candidate vectors by id so SimLens drops into any stack.

Only an in-memory adapter ships by default (zero deps). Qdrant/pgvector/FAISS adapters
follow the same `VectorStore` protocol and are thin optional extras.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class VectorStore(Protocol):
    def get(self, ids: list) -> np.ndarray:  # [len(ids), dim]
        ...


class Memory:
    """Trivial in-memory id -> vector store."""

    def __init__(self, vectors: dict | None = None):
        self._v: dict = {}
        if vectors:
            for k, v in vectors.items():
                self.add(k, v)

    def add(self, id, vector) -> "Memory":
        self._v[id] = np.asarray(vector, dtype=np.float32).ravel()
        return self

    def get(self, ids: list) -> np.ndarray:
        return np.stack([self._v[i] for i in ids])

    def __len__(self) -> int:
        return len(self._v)


__all__ = ["VectorStore", "Memory"]
