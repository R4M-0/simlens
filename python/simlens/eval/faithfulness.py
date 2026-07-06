"""Faithfulness metrics: completeness residuals, deletion curves, scorecards."""
from __future__ import annotations

import numpy as np

from ..explain import Explainer


def faithfulness(explainer: Explainer, pairs: list[tuple], level: str | None = None) -> dict:
    """Aggregate completeness residual + coverage over (query, candidate) pairs."""
    residuals, coverages, rel = [], [], []
    for q, c in pairs:
        a = explainer.explain(q, c, level=level, top_k=10_000)
        residuals.append(a.completeness_residual)
        coverages.append(a.coverage)
        rel.append(a.completeness_residual / (abs(a.score) or 1.0))
    return {
        "n": len(pairs),
        "level": level or explainer._default_level(),
        "residual_mean": float(np.mean(residuals)) if residuals else 0.0,
        "residual_max": float(np.max(residuals)) if residuals else 0.0,
        "relative_residual_mean": float(np.mean(rel)) if rel else 0.0,
        "coverage_mean": float(np.mean(coverages)) if coverages else 1.0,
    }


def deletion_curve(explainer: Explainer, query, candidate, level: str | None = None) -> dict:
    """Delete top-attributed dims from the query in order; a faithful ranking makes the
    score fall faster than a random deletion order (lower AUC = more faithful)."""
    q = np.asarray(query, dtype=np.float64).copy()
    c = np.asarray(candidate, dtype=np.float64)
    a = explainer.explain(query, candidate, level="dim", top_k=10_000)
    order = [int(con.id.split(":")[1]) for con in a.contributions]

    def curve(idx_order):
        qq = q.copy()
        scores = [float(explainer.explain(qq, c, level="dim").score)]
        for i in idx_order:
            qq[i] = 0.0
            scores.append(float(explainer.explain(qq, c, level="dim").score))
        return scores

    rng = np.random.default_rng(0)
    rand = list(order)
    rng.shuffle(rand)
    top = curve(order[: min(len(order), 20)])
    base = curve(rand[: min(len(rand), 20)])
    return {
        "deletion_scores_top": top,
        "deletion_scores_random": base,
        "auc_top": float(np.trapezoid(top)),
        "auc_random": float(np.trapezoid(base)),
        "faithful": float(np.trapezoid(top)) < float(np.trapezoid(base)),
    }


def scorecard(explainer: Explainer, pairs: list[tuple]) -> dict:
    """A compact faithfulness scorecard across available levels."""
    out = {"bundle_hash": explainer._hash}
    for lvl in ["dim", "feature", "concept"]:
        try:
            out[lvl] = faithfulness(explainer, pairs, level=lvl)
        except ValueError:
            out[lvl] = None
    return out
