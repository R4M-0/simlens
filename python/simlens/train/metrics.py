"""Quality metrics for SAE dictionaries — the numbers that decide whether a bundle is
good enough to trust. Reused by the trainer (T1.1) and bundle certification (T2.3).

All metrics operate on plain numpy so they have no heavy dependencies and can run at
build time.
"""
from __future__ import annotations

import numpy as np


def fvu(X: np.ndarray, recon: np.ndarray) -> float:
    """Fraction of variance unexplained = ‖x−x̂‖² / ‖x−x̄‖². Lower is better (0 = perfect)."""
    X = np.asarray(X, dtype=np.float64)
    recon = np.asarray(recon, dtype=np.float64)
    num = float(((X - recon) ** 2).sum())
    den = float(((X - X.mean(axis=0)) ** 2).sum())
    return num / den if den > 0 else 0.0


def l0(H: np.ndarray) -> float:
    """Mean number of active (nonzero) latents per row — the realized sparsity."""
    H = np.asarray(H)
    return float((H != 0).sum(axis=1).mean()) if H.size else 0.0


def dead_fraction(H: np.ndarray) -> float:
    """Fraction of latents that never fire over the sample (a dictionary-health check)."""
    H = np.asarray(H)
    if H.size == 0:
        return 0.0
    ever = (H != 0).any(axis=0)
    return float(1.0 - ever.mean())


def _unit_rows(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


def _ndcg_at_k(ranked_relevance: np.ndarray, ideal_relevance: np.ndarray, k: int) -> float:
    def dcg(rel):
        rel = rel[:k]
        discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
        return float((rel * discounts).sum())
    idcg = dcg(np.sort(ideal_relevance)[::-1])
    return dcg(ranked_relevance) / idcg if idcg > 0 else 0.0


def rank_preservation(
    X: np.ndarray,
    recon: np.ndarray,
    k: int = 10,
    n_queries: int = 128,
    seed: int = 0,
) -> float:
    """nDCG@k of reconstruction-space retrieval against raw-space ground-truth neighbors.

    This is the metric that actually matters for a *retrieval* explainer: if we search
    with the SAE's reconstructions, do we recover (in order) the neighbors the raw space
    would have returned? 1.0 = order perfectly preserved.
    """
    X = _unit_rows(np.asarray(X, dtype=np.float64))
    R = _unit_rows(np.asarray(recon, dtype=np.float64))
    n = X.shape[0]
    if n <= k + 1:
        return 1.0
    rng = np.random.default_rng(seed)
    qs = rng.choice(n, size=min(n_queries, n), replace=False)
    scores = []
    for i in qs:
        raw = X @ X[i]
        raw[i] = -np.inf
        gt = np.argsort(raw)[::-1][:k]  # raw-space top-k (ground truth)
        # graded relevance: k for the top gt neighbor down to 1 for the k-th
        rel = {int(g): float(k - r) for r, g in enumerate(gt)}
        rec = R @ R[i]
        rec[i] = -np.inf
        order = np.argsort(rec)[::-1][:k]  # recon-space top-k
        ranked = np.array([rel.get(int(j), 0.0) for j in order])
        ideal = np.array(list(rel.values()))
        scores.append(_ndcg_at_k(ranked, ideal, k))
    return float(np.mean(scores)) if scores else 1.0


def sae_quality(sae, X: np.ndarray, k: int = 10) -> dict:
    """Full SAE quality scorecard: FVU, L0, dead %, rank-preservation nDCG@k."""
    X = np.asarray(X, dtype=np.float64)
    H = sae.encode(X)
    recon = sae.decode(H)
    return {
        "fvu": round(fvu(X, recon), 6),
        "l0": round(l0(H), 4),
        "dead_pct": round(100.0 * dead_fraction(H), 4),
        "rank_preservation_ndcg": round(rank_preservation(X, recon, k=k), 6),
        "n_features": int(sae.n_features),
        "n_sample": int(X.shape[0]),
    }
