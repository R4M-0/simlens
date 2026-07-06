"""Hierarchical / compositional concepts (frontier) — roll fine SAE features into higher
concepts by feature co-activation clustering (a lightweight meta-SAE).

Fine features are often facets of one higher idea ("these 5 features = kinase inhibitor").
We cluster features whose activations co-occur across the corpus, then register each cluster
as a *meta-concept* whose direction is the (normalized) sum of its members' decoder atoms —
so the concept level gains a coarser, more human layer without retraining.
"""
from __future__ import annotations

import numpy as np

from .bundle import Bundle
from .eval.certify import _encode


def _cooccurrence(H: np.ndarray, min_fire: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Correlation of binarized feature activations (which features fire together)."""
    A = (H > 0).astype(np.float64)
    fires = A.sum(axis=0)
    live = np.where(fires >= min_fire)[0]
    if live.size < 2:
        return np.zeros((0, 0)), live
    B = A[:, live]
    B = B - B.mean(axis=0)
    norm = np.linalg.norm(B, axis=0)
    norm[norm == 0] = 1.0
    Bn = B / norm
    return Bn.T @ Bn, live


def _cluster(corr: np.ndarray, threshold: float) -> list[list[int]]:
    """Connected components over the co-occurrence graph (edges where corr ≥ threshold)."""
    n = corr.shape[0]
    seen = np.zeros(n, dtype=bool)
    clusters = []
    for i in range(n):
        if seen[i]:
            continue
        stack, comp = [i], []
        seen[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.where((corr[u] >= threshold) & ~seen)[0]:
                seen[v] = True
                stack.append(int(v))
        clusters.append(comp)
    return clusters


def discover_hierarchy(
    bundle: Bundle,
    vectors: np.ndarray,
    threshold: float = 0.4,
    min_size: int = 2,
    max_meta: int = 32,
    min_fire: int = 5,
) -> list[dict]:
    """Discover meta-concepts from feature co-activation and register them on ``bundle``.

    Returns a list of ``{"name", "members", "feature_indices"}`` and adds each as a concept
    (``source="hierarchy"``, ``aspect="meta"``). Directions are the normalized sum of member
    decoder atoms, so they live in the same space as the embeddings.
    """
    if not bundle.has_sae:
        raise ValueError("hierarchy needs an SAE bundle")
    X = np.asarray(vectors, dtype=np.float64)
    H = _encode(bundle, X)
    corr, live = _cooccurrence(H, min_fire=min_fire)
    if live.size < 2:
        return []
    w_dec = np.asarray(bundle.w_dec, dtype=np.float64)  # [n_features, dim]
    names = bundle.feature_names

    metas = []
    for comp in _cluster(corr, threshold):
        if len(comp) < min_size:
            continue
        feats = [int(live[c]) for c in comp]
        member_names = [names[f] for f in feats if f < len(names) and names[f]]
        # rank clusters by total activation mass so the strongest surface first
        mass = float(H[:, feats].sum())
        direction = w_dec[feats].sum(axis=0)
        nrm = np.linalg.norm(direction)
        if nrm == 0:
            continue
        direction = direction / nrm
        label = " + ".join(dict.fromkeys(member_names)) if member_names else f"meta[{len(feats)} features]"
        metas.append({"name": label, "members": member_names, "feature_indices": feats,
                     "_mass": mass, "_dir": direction.astype(np.float32)})

    metas.sort(key=lambda m: m["_mass"], reverse=True)
    metas = metas[:max_meta]
    for m in metas:
        name = m["name"]
        if name in bundle.concept_names:
            name = f"{name} (meta)"
        if bundle.concept_dirs is None:
            bundle.concept_dirs = m["_dir"].reshape(1, -1)
        else:
            bundle.concept_dirs = np.vstack([bundle.concept_dirs, m["_dir"].reshape(1, -1)])
        bundle.concept_names.append(name)
        bundle.concept_conf.append(0.0)
        bundle.concept_source.append("hierarchy")
        bundle.aspects[name] = "meta"
        m["name"] = name
    for m in metas:
        m.pop("_mass", None)
        m.pop("_dir", None)
    return metas
