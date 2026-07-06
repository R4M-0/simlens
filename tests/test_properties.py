"""Property-based tests (§6 quality floor) — the completeness guarantees deserve generative
tests, not just examples. Hypothesis searches for a counterexample to each invariant."""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import simlens
from simlens import LearnedMetricExplainer

# bounded finite floats so the arithmetic stays well-conditioned
_floats = st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)


def _vecs(dim):
    return arrays(np.float32, (dim,), elements=_floats)


@settings(max_examples=60, deadline=None)
@given(dim=st.integers(2, 40), data=st.data())
@pytest.mark.parametrize("metric", ["dot", "cosine", "euclidean"])
def test_level1_completeness(metric, dim, data):
    q = data.draw(_vecs(dim))
    c = data.draw(_vecs(dim))
    ex = simlens.Explainer(metric=metric)
    a = ex.explain(q, c, level="dim", top_k=10_000)
    assert a.completeness_residual < 1e-3
    assert abs(sum(x.value for x in a.contributions) - a.score) < 1e-3


@settings(max_examples=60, deadline=None)
@given(dim=st.integers(2, 40), data=st.data())
@pytest.mark.parametrize("metric", ["dot", "cosine", "euclidean"])
def test_margin_additivity(metric, dim, data):
    q = data.draw(_vecs(dim)); better = data.draw(_vecs(dim)); worse = data.draw(_vecs(dim))
    ex = simlens.Explainer(metric=metric)
    m = ex.explain_margin(q, better, worse, level="dim", top_k=10_000)
    expect = simlens.score(q.tolist(), better.tolist(), metric) - \
        simlens.score(q.tolist(), worse.tolist(), metric)
    assert m.score == pytest.approx(expect, abs=1e-3)
    assert abs(sum(x.value for x in m.contributions) - m.score) < 1e-3


@settings(max_examples=40, deadline=None)
@given(dim=st.integers(2, 30), data=st.data())
@pytest.mark.parametrize("metric", ["dot", "cosine", "euclidean"])
def test_metric_symmetry(metric, dim, data):
    q = data.draw(_vecs(dim)); c = data.draw(_vecs(dim))
    assert simlens.score(q.tolist(), c.tolist(), metric) == pytest.approx(
        simlens.score(c.tolist(), q.tolist(), metric), abs=1e-4)


@settings(max_examples=40, deadline=None)
@given(dim=st.integers(2, 20), data=st.data())
@pytest.mark.parametrize("metric", ["dot", "cosine", "euclidean"])
def test_permutation_invariance(metric, dim, data):
    q = data.draw(_vecs(dim)); c = data.draw(_vecs(dim))
    perm = np.random.default_rng(int(q.sum() * 1000) % 2**31).permutation(dim)
    s0 = simlens.score(q.tolist(), c.tolist(), metric)
    s1 = simlens.score(q[perm].tolist(), c[perm].tolist(), metric)
    assert s0 == pytest.approx(s1, abs=1e-4)


@settings(max_examples=40, deadline=None)
@given(dim=st.integers(2, 24), data=st.data())
def test_ig_completeness_linear(dim, data):
    q = data.draw(_vecs(dim)); c = data.draw(_vecs(dim))
    scorer = lambda a, b: float(np.dot(a, b))
    ig = LearnedMetricExplainer(scorer, baseline="zero", steps=8)
    a = ig.explain(q, c, top_k=dim)
    # linear scorer ⇒ IG completeness exact regardless of steps
    assert a.completeness_residual < 1e-3 * (abs(a.score) + 1.0)


@settings(max_examples=30, deadline=None)
@given(m=st.integers(1, 6), n=st.integers(1, 6), dim=st.integers(2, 16), data=st.data())
def test_maxsim_completeness(m, n, dim, data):
    Q = data.draw(arrays(np.float64, (m, dim), elements=_floats))
    C = data.draw(arrays(np.float64, (n, dim), elements=_floats))
    a = simlens.MultiVectorExplainer("dot").explain(Q, C)
    assert abs(a.score - sum(x.value for x in a.contributions)) < 1e-6
