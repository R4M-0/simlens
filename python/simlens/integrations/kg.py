"""Knowledge-graph / knowledge-generation extension.

Turns raw vector similarity into *typed, explained, auditable* edges, and can propose a
concept-annotated subgraph from a set of node vectors. Business-logic-agnostic: you supply
node vectors and (optionally) an `edge_formatter`; you map the returned dicts onto your own
graph schema.
"""
from __future__ import annotations

import numpy as np

from .._native import score as _score
from ..explain import Explainer


def _default_format(concepts: list[dict]) -> str:
    # a concept reads as "shared" only when both nodes actually exhibit it (polarity
    # shared) and it contributes positively — not merely "both lack it"
    named = [
        c["name"]
        for c in concepts
        if c.get("name")
        and c.get("weight", 0) > 0
        and c.get("polarity") == "shared"
        and not str(c["name"]).startswith(("feat:", "dim:", "(bias"))
    ]
    if not named:
        return "Similar (no high-confidence concept mapping)"
    return "Similar: shared " + " & ".join(named[:3])


class KnowledgeGraphExplainer:
    def __init__(self, bundle_or_explainer, level: str | None = None, top_k: int = 5,
                 edge_formatter=None, center: bool | None = None):
        self.ex = bundle_or_explainer if isinstance(bundle_or_explainer, Explainer) else Explainer(bundle_or_explainer)
        self.level = level or self.ex.preferred_level()
        self.top_k = top_k
        self.fmt = edge_formatter or _default_format
        self.center = self.ex.has_centering if center is None else center

    # ---- edges ---------------------------------------------------------------
    def explain_edge(self, a, b, level: str | None = None) -> dict:
        """A domain-explained similarity edge between two node vectors."""
        level = level or self.level
        attr = self.ex.explain(a, b, level=level, top_k=self.top_k, center=self.center)
        dims = self.ex.explain(a, b, level="dim", top_k=self.top_k)
        concepts = [
            {
                "name": c.name,
                "weight": round(c.value, 4),
                "confidence": c.confidence,
                "polarity": c.polarity,
                "evidence": c.evidence,
                "source": c.source,
            }
            for c in attr.contributions
        ]
        conf = next((c.confidence for c in attr.contributions if c.confidence is not None), None)
        return {
            "correlationType": "similar",
            "strength": round(attr.score, 4),
            "explanation": self.fmt(concepts),
            "concepts": concepts,
            "contributingDimensions": [
                int(c.id.split(":")[1]) for c in dims.contributions if c.id.startswith("dim:")
            ],
            "confidence": conf,
            "completenessResidual": round(attr.completeness_residual, 4),
            "warnings": attr.warnings,
        }

    def label_edge(self, a, b) -> str:
        return self.ex.explain(a, b, level=self.level, top_k=self.top_k).as_sentence()

    def why_ranked(self, query, better, worse, level: str | None = None) -> dict:
        m = self.ex.explain_margin(query, better, worse, level=level or self.level, top_k=self.top_k)
        return {"margin": round(m.score, 4), "explanation": m.as_sentence(),
                "contributions": [{"name": c.label, "weight": round(c.value, 4)} for c in m.contributions]}

    # ---- subgraph generation -------------------------------------------------
    def propose_edges(self, nodes: dict, threshold: float = 0.6, level: str | None = None) -> list[dict]:
        """All node pairs above `threshold` become explained, typed edges.

        `nodes` maps id -> vector. O(n²) exhaustive; for larger node sets prefer
        `propose_edges_knn`, which is O(n·k).
        """
        ids = list(nodes)
        vecs = {i: np.asarray(nodes[i], np.float32).ravel().tolist() for i in ids}
        edges = []
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                a, b = ids[x], ids[y]
                s = _score(vecs[a], vecs[b], self.ex.metric)
                if s < threshold:
                    continue
                e = self.explain_edge(vecs[a], vecs[b], level=level)
                top = next(
                    (c["name"] for c in e["concepts"]
                     if c.get("name") and not str(c["name"]).startswith(("(bias", "feat:", "dim:"))),
                    "similar",
                )
                edges.append({"source": a, "target": b, "type": top, **e})
        return edges

    def propose_edges_knn(
        self,
        nodes: dict,
        k: int = 10,
        threshold: float = 0.6,
        level: str | None = None,
    ) -> list[dict]:
        """kNN candidate generation → explained edges in O(n·k) instead of O(n²).

        Each node's top-``k`` neighbours (by the bundle's metric) become candidate edges;
        only those above ``threshold`` are explained. Uses ``hnswlib`` when installed for
        sublinear search, else a vectorized numpy top-k. Edges are de-duplicated.
        """
        ids = list(nodes)
        X = np.asarray([np.asarray(nodes[i], np.float32).ravel() for i in ids], np.float32)
        n = len(ids)
        if n <= 1:
            return []
        k = min(k, n - 1)
        neighbors = self._knn(X, k)

        seen: set = set()
        edges = []
        for xi in range(n):
            for yi in neighbors[xi]:
                if yi == xi:
                    continue
                key = (xi, yi) if xi < yi else (yi, xi)
                if key in seen:
                    continue
                seen.add(key)
                a, b = ids[key[0]], ids[key[1]]
                s = _score(X[key[0]].tolist(), X[key[1]].tolist(), self.ex.metric)
                if s < threshold:
                    continue
                e = self.explain_edge(X[key[0]], X[key[1]], level=level)
                top = next(
                    (c["name"] for c in e["concepts"]
                     if c.get("name") and not str(c["name"]).startswith(("(bias", "feat:", "dim:"))),
                    "similar",
                )
                edges.append({"source": a, "target": b, "type": top, **e})
        return edges

    def _knn(self, X: np.ndarray, k: int) -> np.ndarray:
        """Top-k neighbour indices per row (excluding self)."""
        cosine = self.ex.metric == "cosine"
        Q = X.astype(np.float64)
        if cosine:
            Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
        try:  # sublinear ANN when available
            import hnswlib

            space = "cosine" if cosine else ("ip" if self.ex.metric == "dot" else "l2")
            idx = hnswlib.Index(space=space, dim=X.shape[1])
            idx.init_index(max_elements=len(X), ef_construction=200, M=16)
            idx.add_items(Q, np.arange(len(X)))
            idx.set_ef(max(k + 1, 50))
            labels, _ = idx.knn_query(Q, k=k + 1)
            return labels
        except ImportError:
            pass
        # exact numpy fallback (still O(n²) memory but vectorized; for modest n)
        if self.ex.metric == "euclidean":
            sq = (Q ** 2).sum(1)
            d = sq[:, None] + sq[None, :] - 2 * Q @ Q.T
            np.fill_diagonal(d, np.inf)
            return np.argpartition(d, k, axis=1)[:, :k]
        S = Q @ Q.T
        np.fill_diagonal(S, -np.inf)
        return np.argpartition(-S, k, axis=1)[:, :k]

    # ---- concept layer -------------------------------------------------------
    def concept_profile(self, vector) -> list[dict]:
        """A single node's concept memberships (projection onto each concept direction)."""
        b = self.ex.bundle
        if b is None or not b.has_concepts:
            return []
        v = np.asarray(vector, np.float64).ravel()
        n = np.linalg.norm(v)
        if self.ex.metric == "cosine" and n > 0:
            v = v / n
        dirs = np.asarray(b.concept_dirs, np.float64)
        scores = dirs @ v
        order = np.argsort(np.abs(scores))[::-1]
        return [{"concept": b.concept_names[k], "score": round(float(scores[k]), 4)} for k in order]

    def concept_layer(self, nodes: dict, top_k: int = 3, min_score: float = 0.0) -> list[dict]:
        """Bipartite entity→concept edges (concepts as first-class graph nodes)."""
        edges = []
        for nid, vec in nodes.items():
            for c in self.concept_profile(vec)[:top_k]:
                if c["score"] > min_score:
                    edges.append({"source": nid, "target": f"concept:{c['concept']}",
                                  "type": "exhibits", "strength": c["score"]})
        return edges
