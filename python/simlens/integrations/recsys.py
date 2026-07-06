"""Recommender extension: faithful "because you liked…" reasons + steerable controls.

Works on any item embeddings. A user is represented by one or more liked-item vectors
(averaged into a profile). Business-logic-agnostic.
"""
from __future__ import annotations

import numpy as np

from ..explain import Explainer
from ._util import reasons_sentence


def _profile(liked) -> np.ndarray:
    a = np.asarray(liked, np.float32)
    return a.mean(axis=0) if a.ndim == 2 else a


class RecsysExplainer:
    def __init__(self, bundle_or_explainer, level: str | None = None, top_k: int = 5,
                 center: bool | None = None):
        self.ex = bundle_or_explainer if isinstance(bundle_or_explainer, Explainer) else Explainer(bundle_or_explainer)
        self.level = level or self.ex.preferred_level()
        self.top_k = top_k
        self.center = self.ex.has_centering if center is None else center

    def because(self, liked, item) -> dict:
        """Why `item` is recommended given the user's liked item(s)."""
        a = self.ex.explain(_profile(liked), item, level=self.level, top_k=self.top_k, center=self.center)
        return {"score": round(a.score, 4), "why": reasons_sentence(a),
                "concepts": [{"name": c.label, "weight": round(c.value, 4),
                              "confidence": c.confidence, "evidence": c.evidence} for c in a.contributions]}

    def why_ranked_above(self, liked, item_a, item_b) -> dict:
        m = self.ex.explain_margin(_profile(liked), item_a, item_b, level=self.level, top_k=self.top_k)
        return {"margin": round(m.score, 4), "why": m.as_sentence()}

    def steer_by_concepts(self, seed, weights: dict) -> np.ndarray:
        """'More X, less Y' in concept space (e.g. {"sports": -1.0, "finance": 0.5})."""
        return self.ex.steer(seed, weights)

    def nudge(self, seed, more=None, less=None, alpha: float = 1.0) -> np.ndarray:
        """'More like these, less like those' using example items instead of concepts."""
        v = np.asarray(seed, np.float32).ravel().copy()
        if more is not None and len(more):
            v = v + alpha * _profile(more).ravel()
        if less is not None and len(less):
            v = v - alpha * _profile(less).ravel()
        return v
