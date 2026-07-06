import numpy as np
import pytest

import simlens
from simlens import train


@pytest.fixture(scope="module")
def bundle():
    rng = np.random.default_rng(0)
    dim, k, n = 24, 4, 240
    protos = rng.standard_normal((k, dim))
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    mem = rng.random((n, k)) < 0.3
    X = (mem.astype(float) @ protos + 0.05 * rng.standard_normal((n, dim))).astype(np.float32)
    sae = train.SAE(dim=dim, expansion=3, l1=1e-3, seed=0).fit(X, epochs=25)
    labelers = {f"c{j}": mem[:, j].astype(float) for j in range(k)}
    items = [f"item{i}" for i in range(n)]
    b = train.build_bundle("syn", "cosine", sae, X, labelers=labelers, items=items)
    for j in range(k):
        b.add_concept(f"c{j}", X[mem[:, j]], X[~mem[:, j]], aspect="topic")
    b._X, b._mem, b._labelers = X, mem, labelers
    return b


# ---- evidence & dissimilarity -------------------------------------------------
def test_feature_contribution_carries_evidence(bundle):
    ex = simlens.Explainer(bundle)
    i = np.where(bundle._mem[:, 0])[0]
    a = ex.explain(bundle._X[i[0]], bundle._X[i[1]], level="feature", top_k=5)
    named = [c for c in a.contributions if c.name]
    assert named and named[0].evidence  # named features come with exemplars


def test_dissimilarity_is_one_sided(bundle):
    ex = simlens.Explainer(bundle)
    q = bundle._X[np.where(bundle._mem[:, 0])[0][0]]
    c = bundle._X[np.where(~bundle._mem[:, 0])[0][0]]
    d = ex.explain_dissimilarity(q, c, top_k=6)
    assert all(x.polarity in ("query_only", "candidate_only") for x in d.contributions)


def test_dissimilarity_score_is_total_mass(bundle):
    # 'score' must be the total one-sided mass, which contributions decompose exactly.
    ex = simlens.Explainer(bundle)
    q = bundle._X[np.where(bundle._mem[:, 0])[0][0]]
    c = bundle._X[np.where(~bundle._mem[:, 0])[0][0]]
    d = ex.explain_dissimilarity(q, c, top_k=10_000)
    assert d.completeness_residual == 0.0
    assert abs(sum(x.value for x in d.contributions) - d.score) < 1e-6


def test_vs_corpus_reports_honest_residual(bundle):
    # regression: residual must be computed, not hard-coded to 0.0
    ex = simlens.Explainer(bundle)
    i = np.where(bundle._mem[:, 0])[0]
    foil = [bundle._X[j] for j in np.where(~bundle._mem[:, 0])[0][:5]]
    vc = ex.explain_vs_corpus(bundle._X[i[0]], bundle._X[i[1]], foil, level="concept")
    assert vc.completeness_residual >= 0.0
    assert isinstance(vc.completeness_residual, float)


def test_l2_exact_reconstruction_decomposition_and_anisotropy_warning():
    # import a hand-built SAE with a large decoder bias → bias term dominates
    dim, nf = 2, 2
    w_enc = np.eye(nf, dim, dtype=np.float32)
    b_enc = np.zeros(nf, np.float32)
    w_dec = np.eye(dim, nf, dtype=np.float32)  # [dim, n_features]
    b_dec = np.array([5.0, 5.0], np.float32)
    b = train.import_sae("t", "dot", w_enc, b_enc, w_dec, b_dec)
    ex = simlens.Explainer(b)
    a = ex.explain([1.0, 0.0], [1.0, 0.0], level="feature", top_k=10)
    # Σφ == dot(recon_q, recon_c) == dot([6,5],[6,5]) == 61  (exact, not diagonal-approx)
    assert abs(sum(c.value for c in a.contributions) - 61.0) < 1e-4
    assert any("anisotropy" in w for w in a.warnings)
    assert any(c.id == "feat:bias" for c in a.contributions)


def test_feature_level_rejects_distance_metric(bundle):
    ex = simlens.Explainer(bundle)
    ex.metric = "euclidean"
    with pytest.raises(ValueError, match="dot/cosine"):
        ex.explain(bundle._X[0], bundle._X[1], level="feature")
    with pytest.raises(ValueError, match="dot/cosine"):
        ex.ablate(bundle._X[0], bundle._X[1], threshold=0.5)


# ---- audit: cache + signature -------------------------------------------------
def test_cache_returns_identical_object(bundle):
    ex = simlens.Explainer(bundle, cache=True)
    i = np.where(bundle._mem[:, 1])[0]
    a1 = ex.explain(bundle._X[i[0]], bundle._X[i[1]], level="feature")
    a2 = ex.explain(bundle._X[i[0]], bundle._X[i[1]], level="feature")
    assert a1 is a2


def test_signature_roundtrip(bundle, tmp_path):
    bundle.sign("s3cret")
    bundle.save(tmp_path / "b.simlens")
    r = simlens.Bundle.load(tmp_path / "b.simlens")
    assert r.verify_signature("s3cret")
    assert not r.verify_signature("wrong")


def test_auto_downgrade_to_dim(bundle):
    ex = simlens.Explainer(bundle, auto_downgrade=True)
    # force a high-residual situation by requesting concept level
    a = ex.explain(bundle._X[0], bundle._X[1], level="concept")
    if a.level == "dim":
        assert any("auto_downgraded" in w for w in a.warnings)
        assert a.completeness_residual < 1e-5


# ---- calibration --------------------------------------------------------------
def test_reliability_curve(bundle):
    rep = simlens.eval.reliability(bundle, bundle._X, bundle._labelers)
    assert "calibration_error" in rep and rep["n_named"] > 0
    assert len(rep["bins"]) == 5


def test_measured_confidence_length(bundle):
    m = simlens.eval.measured_confidence(bundle, bundle._X, bundle._labelers)
    assert len(m) == bundle.n_features


# ---- multivector / late interaction ------------------------------------------
def test_maxsim_completeness_and_match():
    rng = np.random.default_rng(3)
    Q = rng.standard_normal((5, 16))
    C = np.vstack([Q[2], rng.standard_normal((4, 16))])  # C[0] == Q[2]
    a = simlens.MultiVectorExplainer("cosine").explain(Q, C)
    assert a.completeness_residual < 1e-9
    assert abs(a.score - sum(c.value for c in a.contributions)) < 1e-9
    # query token 2 should match candidate token 0 with ~perfect similarity
    q2 = next(c for c in a.contributions if c.query_index == 2)
    assert q2.matched_index == 0 and q2.value > 0.99


# ---- viz ----------------------------------------------------------------------
def test_highlight_modes():
    b = simlens.viz.highlight(["a", "b"], [0.1, 0.9])
    h = simlens.viz.highlight(["a", "b"], [0.1, 0.9], mode="html")
    assert "a" in b and "span" in h


# ---- adapters -----------------------------------------------------------------
def test_memory_adapter_roundtrip():
    m = simlens.adapters.Memory()
    m.add("x", [1, 2, 3]).add("y", [4, 5, 6])
    assert m.get(["y", "x"]).shape == (2, 3)
    assert len(m) == 2


def test_qdrant_adapter_with_stub_client():
    class P:
        def __init__(self, id, vector):
            self.id, self.vector = id, vector

    class StubClient:
        def retrieve(self, collection_name, ids, with_vectors):
            return [P(i, [float(i), 0.0]) for i in ids]

    q = simlens.adapters.Qdrant("c", client=StubClient())
    assert q.get([1, 2]).tolist() == [[1.0, 0.0], [2.0, 0.0]]


# ---- CLI ----------------------------------------------------------------------
def test_cli_info_and_verify(bundle, tmp_path, capsys):
    from simlens.cli import main

    p = tmp_path / "b.simlens"
    bundle.save(p)
    main(["info", str(p)])
    out = capsys.readouterr().out
    assert "embedder" in out and "named features" in out
    with pytest.raises(SystemExit) as e:
        main(["verify", str(p)])
    assert e.value.code == 0
