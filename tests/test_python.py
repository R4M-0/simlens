import numpy as np
import pytest

import simlens
from simlens import train


# ---------- Level 1: exact -----------------------------------------------------
@pytest.mark.parametrize("metric", ["dot", "cosine", "euclidean"])
def test_level1_completeness_exact(metric):
    rng = np.random.default_rng(1)
    q = rng.standard_normal(64).astype(np.float32)
    c = rng.standard_normal(64).astype(np.float32)
    ex = simlens.Explainer(metric=metric)
    a = ex.explain(q, c, top_k=10_000)
    assert a.completeness_residual < 1e-5
    assert abs(sum(x.value for x in a.contributions) - a.score) < 1e-5
    assert a.coverage == pytest.approx(1.0)


def test_cosine_matches_numpy():
    rng = np.random.default_rng(2)
    q = rng.standard_normal(32).astype(np.float32)
    c = rng.standard_normal(32).astype(np.float32)
    expect = float(q @ c / (np.linalg.norm(q) * np.linalg.norm(c)))
    assert simlens.score(q.tolist(), c.tolist(), "cosine") == pytest.approx(expect, abs=1e-6)


def test_default_level_is_dim_without_bundle():
    ex = simlens.Explainer(metric="dot")
    assert ex.explain([1, 2, 3], [1, 0, 1]).level == "dim"


def test_margin_completeness():
    rng = np.random.default_rng(3)
    q = rng.standard_normal(40).astype(np.float32)
    better = rng.standard_normal(40).astype(np.float32)
    worse = rng.standard_normal(40).astype(np.float32)
    ex = simlens.Explainer(metric="dot")
    m = ex.explain_margin(q, better, worse, level="dim", top_k=10_000)
    expect = simlens.score(q.tolist(), better.tolist(), "dot") - simlens.score(
        q.tolist(), worse.tolist(), "dot"
    )
    assert m.score == pytest.approx(expect, abs=1e-5)
    assert m.completeness_residual < 1e-5


# ---------- bundle fixture -----------------------------------------------------
@pytest.fixture(scope="module")
def bundle():
    rng = np.random.default_rng(0)
    dim, k, n = 32, 4, 300
    protos = rng.standard_normal((k, dim))
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    mem = rng.random((n, k)) < 0.3
    X = (mem.astype(float) @ protos + 0.05 * rng.standard_normal((n, dim))).astype(np.float32)
    sae = train.SAE(dim=dim, expansion=3, l1=1e-3, seed=0).fit(X, epochs=30)
    labelers = {f"c{j}": mem[:, j].astype(float) for j in range(k)}
    b = train.build_bundle("synthetic", "cosine", sae, X, labelers=labelers)
    for j in range(k):
        b.add_concept(f"c{j}", X[mem[:, j]], X[~mem[:, j]], aspect="topic")
    b._X, b._mem = X, mem  # stash for tests
    return b


def test_bundle_roundtrip_and_verify(tmp_path, bundle):
    p = tmp_path / "b.simlens"
    bundle.save(p)
    r = simlens.Bundle.load(p)
    assert r.verify()
    assert r.n_features == bundle.n_features
    assert r.concept_names == bundle.concept_names
    assert r.content_hash == bundle.compute_hash()


def test_feature_explanation_reports_residual(bundle):
    ex = simlens.Explainer(bundle)
    idx = np.where(bundle._mem[:, 0])[0]
    a = ex.explain(bundle._X[idx[0]], bundle._X[idx[1]], level="feature", top_k=5)
    assert a.level == "feature"
    assert a.bundle_hash is not None
    valid = {"shared", "query_only", "candidate_only", "neither"}
    assert all(c.polarity in valid for c in a.contributions)
    # residual is defined and finite
    assert a.completeness_residual >= 0.0


def test_concept_explanation_has_partial_warning(bundle):
    ex = simlens.Explainer(bundle)
    a = ex.explain(bundle._X[0], bundle._X[1], level="concept")
    assert a.level == "concept"
    assert any("partial_decomposition" in w for w in a.warnings)


def test_ablation_breaks_match(bundle):
    ex = simlens.Explainer(bundle)
    idx = np.where(bundle._mem[:, 1])[0]
    q, c = bundle._X[idx[0]], bundle._X[idx[1]]
    before = simlens.score(q.tolist(), c.tolist(), "cosine")
    abl = ex.ablate(q, c, threshold=before * 0.5)
    assert abl["score_after"] <= abl["score_before"]
    assert abl["dropped_below"]


def test_steer_moves_projection(bundle):
    ex = simlens.Explainer(bundle)
    q = bundle._X[0]
    q2 = ex.steer(q, {"c1": 3.0})
    assert q2.shape == q.shape
    assert not np.allclose(q2, q)


def test_aspect_grouping(bundle):
    ex = simlens.Explainer(bundle)
    a = ex.explain(bundle._X[0], bundle._X[1], level="aspect")
    assert a.level == "aspect"
    assert all(c.name == "topic" for c in a.contributions)


def test_faithfulness_dim_is_exact(bundle):
    ex = simlens.Explainer(bundle)
    pairs = [(bundle._X[i], bundle._X[i + 1]) for i in range(0, 10, 2)]
    f = simlens.eval.faithfulness(ex, pairs, level="dim")
    assert f["residual_max"] < 1e-4
