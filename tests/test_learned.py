"""T2.1 — Integrated Gradients for learned / non-linear metrics."""
import numpy as np
import pytest

from simlens import LearnedMetricExplainer


def test_ig_completeness_linear_is_exact():
    # a linear scorer → IG is exact at any step count (constant gradient)
    rng = np.random.default_rng(0)
    dim = 16
    q = rng.standard_normal(dim)
    c = rng.standard_normal(dim)
    scorer = lambda a, b: float(np.dot(a, b))
    ex = LearnedMetricExplainer(scorer, baseline="zero", steps=16)
    a = ex.explain(q, c, top_k=dim)
    assert a.completeness_residual < 1e-6
    assert abs(sum(x.value for x in a.contributions) - a.score) < 1e-6


def test_ig_completeness_nonlinear_within_tolerance():
    rng = np.random.default_rng(1)
    dim = 12
    W = rng.standard_normal((dim, dim))
    # a non-linear learned metric: tanh bilinear form
    scorer = lambda a, b: float(np.tanh(a @ W) @ b)
    q = 0.3 * rng.standard_normal(dim)
    c = 0.3 * rng.standard_normal(dim)
    ex = LearnedMetricExplainer(scorer, baseline="zero", steps=128)
    a = ex.explain(q, c, top_k=dim)
    # Σφ ≈ s(q,c) − s(0,c) to a small tolerance
    assert a.completeness_residual < 1e-2 * (abs(a.score) + 1.0)


def test_ig_centroid_baseline_semantics():
    rng = np.random.default_rng(2)
    dim = 8
    mu = rng.standard_normal(dim)
    scorer = lambda a, b: float(np.dot(a, b))
    ex = LearnedMetricExplainer(scorer, baseline="centroid", mean=mu, steps=8)
    q = rng.standard_normal(dim)
    c = rng.standard_normal(dim)
    a = ex.explain(q, c, top_k=dim)
    # Σφ should equal s(q,c) − s(μ,c)
    expect = np.dot(q, c) - np.dot(mu, c)
    assert abs(a.score - expect) < 1e-6
    assert any("baseline='centroid'" in w for w in a.warnings)


def test_ig_requires_mean_for_centroid():
    ex = LearnedMetricExplainer(lambda a, b: float(np.dot(a, b)), baseline="centroid")
    with pytest.raises(ValueError, match="centroid"):
        ex.explain(np.ones(4), np.ones(4))


def test_ig_deletion_beats_random():
    rng = np.random.default_rng(3)
    dim = 20
    # a metric where a few dims dominate → attribution should find them
    w = np.zeros(dim); w[[2, 5, 11]] = [3.0, 2.0, 2.5]
    scorer = lambda a, b: float((a * w) @ (b * w))
    q = rng.standard_normal(dim)
    c = rng.standard_normal(dim)
    ex = LearnedMetricExplainer(scorer, baseline="zero", steps=16)
    dc = ex.deletion_curve(q, c, k=10)
    assert dc["faithful"]


def test_user_supplied_gradient():
    dim = 6
    scorer = lambda a, b: float(np.dot(a, b))
    grad = lambda x, c: np.asarray(c)  # ∂(x·c)/∂x = c
    ex = LearnedMetricExplainer(scorer, grad=grad, baseline="zero", steps=4)
    q = np.arange(dim, dtype=float)
    c = np.ones(dim)
    a = ex.explain(q, c, top_k=dim)
    assert a.completeness_residual < 1e-9
