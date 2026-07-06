"""Qdrant adapter — fetch stored vectors by point id."""
from __future__ import annotations

import numpy as np

from .base import Sample, _require


class Qdrant:
    """Wrap a `qdrant_client.QdrantClient` (or connect from url/host)."""

    def __init__(self, collection: str, client=None, url: str | None = None, **kwargs):
        if client is None:
            qc = _require("qdrant_client")
            client = qc.QdrantClient(url=url, **kwargs) if url else qc.QdrantClient(**kwargs)
        self.client = client
        self.collection = collection

    def get(self, ids: list) -> np.ndarray:
        points = self.client.retrieve(
            collection_name=self.collection, ids=list(ids), with_vectors=True
        )
        by_id = {p.id: p.vector for p in points}
        return np.asarray([by_id[i] for i in ids], dtype=np.float32)

    def sample(self, n: int) -> Sample:
        points, _ = self.client.scroll(
            collection_name=self.collection, limit=n, with_vectors=True, with_payload=True
        )
        vectors = np.asarray([p.vector for p in points], dtype=np.float32)
        payloads = [p.payload or {} for p in points]
        ids = [p.id for p in points]
        return Sample(vectors=vectors, payloads=payloads, ids=ids)
