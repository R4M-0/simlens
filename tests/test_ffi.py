"""T2.4 — zero-copy numpy FFI paths must match the list/bytes paths exactly."""
import numpy as np
import pytest

import simlens
from simlens import _native, train
from simlens.train import build_bundle


def test_score_np_matches_list():
    rng = np.random.default_rng(0)
    q = rng.standard_normal(64).astype(np.float32)
    c = rng.standard_normal(64).astype(np.float32)
    for m in ("dot", "cosine", "euclidean"):
        a = _native.score(q.tolist(), c.tolist(), m)
        b = _native.score_np(np.ascontiguousarray(q), np.ascontiguousarray(c), m)
        assert a == pytest.approx(b, abs=1e-9)


def test_explain_l1_np_matches_list():
    rng = np.random.default_rng(1)
    q = rng.standard_normal(48).astype(np.float32)
    c = rng.standard_normal(48).astype(np.float32)
    d1 = _native.explain_l1(q.tolist(), c.tolist(), "cosine", 10_000, 0.0)
    d2 = _native.explain_l1_np(np.ascontiguousarray(q), np.ascontiguousarray(c), "cosine", 10_000, 0.0)
    assert d1["score"] == pytest.approx(d2["score"], abs=1e-9)
    assert len(d1["contributions"]) == len(d2["contributions"])


def test_encode_batch_matches_per_row():
    rng = np.random.default_rng(2)
    dim, n = 32, 60
    X = rng.standard_normal((n, dim)).astype(np.float32)
    sae = train.fit(X, arch="topk", k=8, expansion=4, epochs=10)
    b = build_bundle("syn", "cosine", sae, X)
    nat = b.to_native_sae()
    batch = nat.encode_batch_np(np.ascontiguousarray(X))
    assert batch.shape == (n, sae.n_features)
    for i in range(0, n, 13):
        row = np.asarray(nat.encode_np(np.ascontiguousarray(X[i])))
        assert np.abs(batch[i] - row).max() < 1e-9


def test_from_numpy_constructor_roundtrip():
    rng = np.random.default_rng(3)
    dim, n = 24, 40
    X = rng.standard_normal((n, dim)).astype(np.float32)
    sae = train.fit(X, arch="topk", k=6, expansion=4, epochs=8)
    b = build_bundle("syn", "cosine", sae, X)
    nat = b.to_native_sae()  # uses PySae.from_numpy
    py = sae.encode(X[:4])
    for i in range(4):
        assert np.abs(np.asarray(nat.encode_np(np.ascontiguousarray(X[i]))) - py[i]).max() < 1e-5
