"""Adversarial / edge cases (§6 quality floor): zero vectors, dead SAEs, empty/oversized
bundles, metric mismatch, non-textual exemplars, wrong-length inputs. Nothing should crash,
NaN, or silently read out of bounds."""
import numpy as np
import pytest

import simlens
from simlens import train
from simlens.naming import KeywordNamer, ValidatedNamer
from simlens.train import build_bundle, import_sae


def test_zero_vectors_do_not_nan():
    ex = simlens.Explainer(metric="cosine")
    a = ex.explain(np.zeros(8), np.zeros(8), level="dim")
    assert not np.isnan(a.score)
    assert a.score == 0.0
    b = ex.explain(np.zeros(8), np.ones(8), level="dim")
    assert not np.isnan(b.score)


def test_dead_sae_is_honest_not_crashing():
    # an SAE whose features never fire (huge negative bias) → reconstruction is just b_dec
    dim, nf = 6, 4
    w_enc = np.eye(nf, dim, dtype=np.float32)
    b_enc = np.full(nf, -1e6, np.float32)
    w_dec = np.eye(dim, nf, dtype=np.float32)
    b = import_sae("dead", "dot", w_enc, b_enc, w_dec)
    ex = simlens.Explainer(b)
    a = ex.explain(np.ones(dim), np.ones(dim), level="feature", top_k=10)
    assert not np.isnan(a.completeness_residual)
    # no live features → only the bias term (or nothing) contributes
    assert all(c.id in ("feat:bias",) or c.polarity == "neither" for c in a.contributions) or \
        len(a.contributions) <= 1


def test_empty_bundle_dim_works_feature_raises():
    b = simlens.Bundle(embedder="empty", dim=5, metric="cosine")
    ex = simlens.Explainer(b)
    assert ex.explain(np.ones(5), np.arange(5.0)).level == "dim"
    with pytest.raises(ValueError, match="no SAE"):
        ex.explain(np.ones(5), np.ones(5), level="feature")


def test_metric_mismatch_rejected():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 12)).astype(np.float32)
    sae = train.fit(X, arch="topk", k=4, expansion=3, epochs=8)
    b = build_bundle("m", "euclidean", sae, X)
    ex = simlens.Explainer(b)
    with pytest.raises(ValueError, match="dot/cosine"):
        ex.explain(X[0], X[1], level="feature")


def test_wrong_length_vector_errors_cleanly():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 10)).astype(np.float32)
    sae = train.fit(X, arch="topk", k=4, expansion=3, epochs=6)
    ex = simlens.Explainer(build_bundle("w", "cosine", sae, X))
    with pytest.raises(ValueError, match="dim"):
        ex.explain(np.ones(11), np.ones(11), level="feature")


def test_non_textual_exemplars_named_gracefully():
    # integer ids, not text → keyword namer returns None rather than nonsense
    name, conf = KeywordNamer().name([1, 2, 3])
    assert name is None and conf == 0.0
    vn = ValidatedNamer(lambda p: "x")
    label, acc = vn.name([1, 2, 3])
    assert acc == 0.0 or label is not None  # never raises


def test_constant_and_singleton_corpus():
    # degenerate corpora must not crash autofit
    X = np.ones((30, 8), dtype=np.float32)
    b = simlens.autofit(vectors=X, embedder="const", metric="cosine", expansion=2, epochs=5,
                        certify=False, center="center")
    assert b.n_features > 0
    single = np.random.default_rng(0).standard_normal((1, 8)).astype(np.float32)
    b2 = simlens.autofit(vectors=single, embedder="one", metric="dot", expansion=2, epochs=3,
                        certify=False, center=None)
    assert b2.n_features > 0


def test_oversized_topk_falls_back_to_relu():
    # k >= n_features must not break the gate (keep all)
    rng = np.random.default_rng(2)
    X = rng.standard_normal((50, 6)).astype(np.float32)
    sae = train.fit(X, arch="topk", k=1000, expansion=2, epochs=5)
    H = sae.encode(X)
    assert H.shape[1] == sae.n_features  # all features available, no crash
