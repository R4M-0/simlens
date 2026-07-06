"""Late-interaction (multi-vector) attribution — §13.10.

For token-level retrievers (ColBERT-style), an item is a *set* of vectors. The MaxSim
score is `Σ_i max_j sim(qᵢ, cⱼ)` — a sum over query tokens, so it decomposes *exactly*:
each query token's contribution is its best match into the candidate, and we know which
candidate token it matched. Completeness holds by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _rownorm(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


@dataclass
class TokenContribution:
    query_index: int
    query_token: str | None
    value: float
    matched_index: int
    matched_token: str | None


@dataclass
class MultiVectorAttribution:
    score: float
    metric: str
    contributions: list[TokenContribution]
    alignment: np.ndarray = field(repr=False)  # [m, n] similarity matrix
    completeness_residual: float = 0.0

    def as_sentence(self, max_terms: int = 3) -> str:
        total = sum(abs(c.value) for c in self.contributions) or 1.0
        top = sorted(self.contributions, key=lambda c: abs(c.value), reverse=True)[:max_terms]
        parts = [
            f"'{c.query_token or c.query_index}'→'{c.matched_token or c.matched_index}' "
            f"({100 * c.value / total:.0f}%)"
            for c in top
        ]
        return "Matched via " + ", ".join(parts) + "."


class MultiVectorExplainer:
    """Explain a MaxSim (late-interaction) similarity between two sets of token vectors."""

    def __init__(self, metric: str = "cosine"):
        self.metric = metric

    def explain(
        self,
        query_tokens: np.ndarray,
        candidate_tokens: np.ndarray,
        query_labels: list | None = None,
        candidate_labels: list | None = None,
    ) -> MultiVectorAttribution:
        Q = np.asarray(query_tokens, dtype=np.float64)
        C = np.asarray(candidate_tokens, dtype=np.float64)
        if self.metric == "cosine":
            Q, C = _rownorm(Q), _rownorm(C)
        S = Q @ C.T  # [m, n]
        best = S.argmax(axis=1)
        maxes = S[np.arange(S.shape[0]), best]
        contribs = [
            TokenContribution(
                query_index=i,
                query_token=query_labels[i] if query_labels else None,
                value=float(maxes[i]),
                matched_index=int(best[i]),
                matched_token=candidate_labels[int(best[i])] if candidate_labels else None,
            )
            for i in range(S.shape[0])
        ]
        contribs.sort(key=lambda c: abs(c.value), reverse=True)
        score = float(maxes.sum())
        residual = abs(score - sum(c.value for c in contribs))
        return MultiVectorAttribution(
            score=score,
            metric=self.metric,
            contributions=contribs,
            alignment=S,
            completeness_residual=residual,
        )
