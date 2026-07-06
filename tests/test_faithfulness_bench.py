"""Faithfulness benchmark harness (§6 quality floor) — runs in CI on a *fixed* bundle to
catch quality regressions. If a change makes the SAE reconstruct worse, the attribution
less complete, or deletion/insertion no longer beat random, this fails."""
import numpy as np
import pytest

import simlens
from simlens import train
from simlens.eval import certify
from simlens.train import build_bundle


@pytest.fixture(scope="module")
def fixed_bundle():
    """A deterministic, well-structured corpus + TopK SAE — the regression baseline."""
    rng = np.random.default_rng(1234)
    dim, kc, n = 64, 8, 1500
    protos = rng.standard_normal((kc, dim))
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    mem = rng.random((n, kc)) < 0.2
    X = (mem @ protos + 0.04 * rng.standard_normal((n, dim))).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)  # cosine embeddings are unit-norm
    sae = train.fit(X, arch="topk", k=24, expansion=8, epochs=40, seed=0)
    bundle = build_bundle("bench", "cosine", sae, X)
    return bundle, X


def test_sae_reconstruction_within_budget(fixed_bundle):
    bundle, X = fixed_bundle
    card = certify(bundle, X, n_pairs=30)
    assert card["sae"]["fvu"] < 0.15, card["sae"]
    assert card["sae"]["l0"] == pytest.approx(24.0, abs=1.0)
    assert card["sae"]["dead_pct"] < 1.0


def _neighbor_pairs(X, n=30):
    """Each query paired with its nearest neighbour — the regime that matters for a
    *retrieval* explainer (random pairs have ~0 cosine, so relative residual is undefined)."""
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    pairs = []
    for i in range(n):
        s = Xn @ Xn[i]; s[i] = -np.inf
        pairs.append((X[i], X[int(s.argmax())]))
    return pairs


def test_attribution_completeness(fixed_bundle):
    bundle, X = fixed_bundle
    ex = simlens.Explainer(bundle)
    pairs = _neighbor_pairs(X, 30)
    dim = simlens.eval.faithfulness(ex, pairs, level="dim")
    feat = simlens.eval.faithfulness(ex, pairs, level="feature")
    assert dim["residual_max"] < 1e-4  # Level-1 is exact
    # Level-2 residual is only SAE reconstruction error. The <0.05 acceptance target is for
    # production-scale bundles (100k passages, tuned); this tiny fixture lands ~0.08, so the
    # regression guard is set a little above that. If it drifts up, reconstruction regressed.
    assert feat["relative_residual_mean"] < 0.12


def test_deletion_and_insertion_beat_random(fixed_bundle):
    bundle, X = fixed_bundle
    card = certify(bundle, X, n_pairs=20)
    di = card["attribution"]["deletion_insertion"]
    assert di["deletion_faithful"]
    assert di["insertion_faithful"]
