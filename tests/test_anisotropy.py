"""T2.2 — anisotropy correction: μ + whitening/ABTT, and the centered "why"."""
import numpy as np
import pytest

import simlens
from simlens.anisotropy import anisotropy_baseline, apply_centering, fit_centering


@pytest.fixture(scope="module")
def anisotropic():
    """A corpus with a big shared mean direction (high anisotropy) + real topic structure."""
    rng = np.random.default_rng(0)
    dim, kc, n = 40, 5, 800
    protos = rng.standard_normal((kc, dim))
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    mem = rng.random((n, kc)) < 0.3
    bias = 6.0 * np.ones(dim)  # dominant shared direction
    X = (bias + mem @ protos + 0.05 * rng.standard_normal((n, dim))).astype(np.float32)
    return X, mem


@pytest.mark.parametrize("mode", ["center", "abtt", "whiten"])
def test_fit_centering_shapes(anisotropic, mode):
    X, _ = anisotropic
    mu, W = fit_centering(X, mode=mode)
    assert mu.shape == (X.shape[1],)
    if mode == "center":
        assert W is None
    else:
        assert W.shape == (X.shape[1], X.shape[1])


def test_centering_reduces_anisotropy(anisotropic):
    X, _ = anisotropic
    raw = anisotropy_baseline(X, metric="cosine")
    mu, W = fit_centering(X, mode="abtt")
    Xc = np.array([apply_centering(x, mu, W) for x in X])
    corrected = anisotropy_baseline(Xc, metric="cosine")
    assert corrected < raw
    assert corrected < 0.5  # from ~0.99 down toward 0


def test_centered_dim_decomposition_is_exact(anisotropic):
    X, _ = anisotropic
    b = simlens.autofit(vectors=X, embedder="syn", metric="cosine", expansion=4, epochs=15,
                        center="abtt")
    ex = simlens.Explainer(b)
    a = ex.explain(X[0], X[1], level="dim", top_k=10_000, center=True)
    assert abs(sum(c.value for c in a.contributions) - a.score) < 1e-4
    assert any("centered" in w for w in a.warnings)


def test_centered_feature_why_surfaces_discriminative(anisotropic):
    X, mem = anisotropic
    b = simlens.autofit(vectors=X, embedder="syn", metric="cosine", expansion=4, epochs=20,
                        center="abtt")
    ex = simlens.Explainer(b)
    # two items sharing topic 0
    i = np.where(mem[:, 0])[0]
    raw = ex.explain(X[i[0]], X[i[1]], level="feature", top_k=10_000)
    centered = ex.explain(X[i[0]], X[i[1]], level="feature", top_k=10_000, center=True)
    # the raw feature attribution is dominated by the bias/global-mean term...
    bias_share_raw = sum(abs(c.value) for c in raw.contributions if c.id == "feat:bias")
    total_raw = sum(abs(c.value) for c in raw.contributions) or 1.0
    # ...while the centered view removes it (no bias term dominating)
    bias_centered = [c for c in centered.contributions if c.id == "feat:bias"]
    assert bias_share_raw / total_raw > 0.2  # baseline really does dominate raw
    assert not bias_centered or abs(bias_centered[0].value) < 0.5 * total_raw
    assert any("centered" in w for w in centered.warnings)


def test_centering_survives_roundtrip(anisotropic, tmp_path):
    X, _ = anisotropic
    b = simlens.autofit(vectors=X, embedder="syn", metric="cosine", expansion=4, epochs=10,
                        center="abtt")
    b.save(tmp_path / "b.simlens")
    r = simlens.Bundle.load(tmp_path / "b.simlens")
    assert r.mean is not None and r.whitening is not None
    assert r.verify()
    assert simlens.Explainer(r).has_centering


def test_integration_defaults_to_centered(anisotropic):
    from simlens.integrations import RagExplainer

    X, _ = anisotropic
    b = simlens.autofit(vectors=X, embedder="syn", metric="cosine", expansion=4, epochs=10,
                        center="abtt")
    rag = RagExplainer(b)
    assert rag.center is True  # auto-enabled because the bundle carries centering
    res = rag.explain_results(X[0], [X[1], X[2]])
    assert len(res) == 2 and "why" in res[0]
