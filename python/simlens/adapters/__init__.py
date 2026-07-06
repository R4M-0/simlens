"""Vector-store adapters: fetch candidate vectors by id so SimLens drops into any stack.

Only the in-memory adapter is dependency-free; the others lazily import their client so
`import simlens.adapters` never fails if a driver isn't installed.
"""
from __future__ import annotations

from .base import Sample, SampleableStore, VectorStore
from .faiss import Faiss
from .memory import Memory
from .pgvector import Pgvector
from .qdrant import Qdrant
from .weaviate import Weaviate

__all__ = [
    "VectorStore",
    "SampleableStore",
    "Sample",
    "Memory",
    "Qdrant",
    "Pgvector",
    "Faiss",
    "Weaviate",
]
