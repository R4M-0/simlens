"""RAG & search extension: explain *why retrieved* and *why ranked*, flag spurious matches.

Business-logic-agnostic: pass a query vector and hits (vectors, or ids + a store to fetch
them). Configure `level`, `top_k`, and an optional `spurious` set of concept names to flag.
"""
from __future__ import annotations

import numpy as np

from ..explain import Explainer
from ._util import reasons_sentence


class RagExplainer:
    def __init__(self, bundle_or_explainer=None, store=None, level: str | None = None,
                 top_k: int = 5, spurious: set | None = None, center: bool | None = None):
        self.ex = bundle_or_explainer if isinstance(bundle_or_explainer, Explainer) else Explainer(bundle_or_explainer)
        self.store = store
        self.level = level or self.ex.preferred_level()
        self.top_k = top_k
        self.spurious = set(spurious or [])
        # default the human-facing "why" to the centered view when the bundle supports it
        self.center = self.ex.has_centering if center is None else center

    def _vecs(self, hits) -> list:
        """Accept hits as vectors, or as ids to fetch from the configured store."""
        first = hits[0] if hits else None
        if isinstance(first, (int, str)):
            if self.store is None:
                raise ValueError("hits are ids but no store was configured")
            return [np.asarray(v, np.float32) for v in self.store.get(list(hits))]
        return [np.asarray(v, np.float32) for v in hits]

    def explain_results(self, query, hits) -> list[dict]:
        """Per-hit: why it matched, whether spurious features drove it, margin vs. the top hit."""
        vecs = self._vecs(hits)
        out = []
        top_vec = vecs[0] if vecs else None
        for rank, v in enumerate(vecs):
            a = self.ex.explain(query, v, level=self.level, top_k=self.top_k, center=self.center)
            flagged = [c.label for c in a.contributions if c.name in self.spurious]
            row = {
                "rank": rank,
                "score": round(a.score, 4),
                "why": reasons_sentence(a),
                "concepts": [{"name": c.label, "weight": round(c.value, 4),
                              "confidence": c.confidence} for c in a.contributions],
                "spurious_flags": flagged,
                "warnings": a.warnings,
            }
            if rank > 0 and top_vec is not None:
                m = self.ex.explain_margin(query, top_vec, v, level=self.level, top_k=self.top_k)
                row["margin_vs_top"] = {"value": round(m.score, 4), "why": m.as_sentence()}
            out.append(row)
        return out

    def why_ranked(self, query, better, worse) -> dict:
        m = self.ex.explain_margin(query, better, worse, level=self.level, top_k=self.top_k)
        return {"margin": round(m.score, 4), "why": m.as_sentence()}

    def why_not(self, query, retrieved, gold) -> dict:
        """Why the retrieved hit beat the gold passage (what the gold lacks / has extra)."""
        m = self.ex.explain_margin(query, retrieved, gold, level=self.level, top_k=self.top_k)
        return {"margin": round(m.score, 4), "why": m.as_sentence(),
                "against_gold": [{"name": c.label, "weight": round(c.value, 4)} for c in m.contributions]}

    def audit_query(self, query, hits) -> dict:
        return {"n_hits": len(hits), "level": self.level or self.ex._default_level(),
                "results": self.explain_results(query, hits)}
