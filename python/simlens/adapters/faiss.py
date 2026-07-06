"""FAISS adapter — reconstruct stored vectors by position (or mapped id)."""
from __future__ import annotations

import numpy as np


class Faiss:
    """Reconstruct vectors from a FAISS index.

    `id_map` optionally maps external ids to internal row positions; without it, ids are
    treated as integer positions. The index must support `reconstruct` (e.g. Flat / IDMap).
    """

    def __init__(self, index, id_map: dict | None = None):
        self.index = index
        self.id_map = id_map

    def get(self, ids: list) -> np.ndarray:
        pos = [self.id_map[i] if self.id_map else int(i) for i in ids]
        return np.asarray([self.index.reconstruct(int(p)) for p in pos], dtype=np.float32)
