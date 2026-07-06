"""Faithfulness certification (T2.3): compute a quality scorecard at *build time* and store
it in the bundle, so a bundle carries a provable, signed quality claim instead of a promise.

The scorecard is written into ``manifest.json`` under ``"faithfulness"`` and is covered by
the content hash (and therefore the signature), so it cannot be altered without detection.
"""
from __future__ import annotations

import numpy as np

from ..bundle import Bundle
from ..train.metrics import dead_fraction, fvu, l0, rank_preservation


def _encode(bundle: Bundle, X: np.ndarray) -> np.ndarray:
    """Gated SAE activations, matching the native kernel (relu → JumpReLU → top-k)."""
    X = np.asarray(X, dtype=np.float64)
    h = np.maximum(X @ np.asarray(bundle.w_enc).T + np.asarray(bundle.b_enc), 0.0)
    if bundle.sae_threshold is not None:
        h = h * (h > np.asarray(bundle.sae_threshold))
    k = int(bundle.sae_k)
    if 0 < k < h.shape[1]:
        idx = np.argpartition(h, -k, axis=1)[:, -k:]
        mask = np.zeros_like(h, dtype=bool)
        mask[np.arange(h.shape[0])[:, None], idx] = True
        h = h * (mask & (h > 0))
    return h


def _decode(bundle: Bundle, H: np.ndarray) -> np.ndarray:
    return np.asarray(H, dtype=np.float64) @ np.asarray(bundle.w_dec) + np.asarray(bundle.b_dec)


def _deletion_insertion_auc(explainer, pairs, k: int = 15) -> dict:
    """Mean deletion/insertion AUC (dim level) vs. random-order baselines (RISE-style)."""
    dels_top, dels_rand, ins_top, ins_rand = [], [], [], []
    for q, c in pairs:
        q = np.asarray(q, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        a = explainer.explain(q, c, level="dim", top_k=10_000)
        order = [int(con.id.split(":")[1]) for con in a.contributions]
        rng = np.random.default_rng(0)
        rand = list(order); rng.shuffle(rand)

        def deletion(idx):
            qq = q.copy(); s = [float(explainer.explain(qq, c, level="dim").score)]
            for i in idx[:k]:
                qq[i] = 0.0; s.append(float(explainer.explain(qq, c, level="dim").score))
            return np.trapezoid(s)

        def insertion(idx):
            qq = np.zeros_like(q); s = [float(explainer.explain(qq, c, level="dim").score)]
            for i in idx[:k]:
                qq[i] = q[i]; s.append(float(explainer.explain(qq, c, level="dim").score))
            return np.trapezoid(s)

        dels_top.append(deletion(order)); dels_rand.append(deletion(rand))
        ins_top.append(insertion(order)); ins_rand.append(insertion(rand))
    return {
        "deletion_auc_top": round(float(np.mean(dels_top)), 6),
        "deletion_auc_random": round(float(np.mean(dels_rand)), 6),
        "insertion_auc_top": round(float(np.mean(ins_top)), 6),
        "insertion_auc_random": round(float(np.mean(ins_rand)), 6),
        # faithful ⇒ deleting important dims hurts more, inserting them helps more
        "deletion_faithful": bool(np.mean(dels_top) < np.mean(dels_rand)),
        "insertion_faithful": bool(np.mean(ins_top) > np.mean(ins_rand)),
    }


def certify(
    bundle: Bundle,
    vectors: np.ndarray,
    n_pairs: int = 40,
    rank_k: int = 10,
    seed: int = 0,
) -> dict:
    """Compute the faithfulness scorecard for ``bundle`` over ``vectors`` and store it on the
    bundle (``bundle.faithfulness``). Returns the scorecard dict."""
    from ..explain import Explainer

    X = np.asarray(vectors, dtype=np.float64)
    card: dict = {"n_sample": int(X.shape[0])}

    if bundle.has_sae:
        H = _encode(bundle, X)
        recon = _decode(bundle, H)
        card["sae"] = {
            "fvu": round(fvu(X, recon), 6),
            "l0": round(l0(H), 4),
            "dead_pct": round(100.0 * dead_fraction(H), 4),
            "rank_preservation_ndcg": round(rank_preservation(X, recon, k=rank_k), 6),
        }

    # attribution faithfulness on a held-out pair sample
    ex = Explainer(bundle)
    rng = np.random.default_rng(seed)
    n = min(2 * n_pairs, X.shape[0] - X.shape[0] % 2)
    idx = rng.choice(X.shape[0], size=n, replace=False)
    pairs = [(X[idx[i]], X[idx[i + 1]]) for i in range(0, n - 1, 2)]
    if pairs:
        from .faithfulness import faithfulness

        card["attribution"] = {
            "dim": faithfulness(ex, pairs, level="dim"),
        }
        if bundle.has_sae:
            card["attribution"]["feature"] = faithfulness(ex, pairs, level="feature")
        card["attribution"]["deletion_insertion"] = _deletion_insertion_auc(ex, pairs[:10])

    # naming: distribution of stored confidences (detection accuracies for AI names)
    confs = [c for c in bundle.feature_conf if isinstance(c, (int, float))]
    named = sum(1 for nprov in bundle.feature_source if nprov)
    if confs:
        card["naming"] = {
            "n_named": named,
            "conf_mean": round(float(np.mean(confs)), 4),
            "conf_median": round(float(np.median(confs)), 4),
            "conf_p10": round(float(np.percentile(confs, 10)), 4),
            "provenance": bundle.name_provenance(),
        }

    card["concepts"] = {"n": len(bundle.concept_names)}
    card["centered"] = bundle.mean is not None
    bundle.faithfulness = card
    return card
