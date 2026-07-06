"""Weaviate adapter — fetch object vectors by uuid."""
from __future__ import annotations

import numpy as np


class Weaviate:
    """Fetch stored vectors from a Weaviate collection by object id.

    Works with the weaviate-client v4 collections API: pass a connected `client` and the
    collection name.
    """

    def __init__(self, client, collection: str):
        self.client = client
        self.collection = collection

    def get(self, ids: list) -> np.ndarray:
        coll = self.client.collections.get(self.collection)
        out = []
        for i in ids:
            obj = coll.query.fetch_object_by_id(i, include_vector=True)
            vec = obj.vector
            if isinstance(vec, dict):  # named vectors → take the default
                vec = vec.get("default") or next(iter(vec.values()))
            out.append(vec)
        return np.asarray(out, dtype=np.float32)
