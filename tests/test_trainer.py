"""T1.1 — the real SAE trainer: TopK/BatchTopK/JumpReLU, quality metrics, and the
train==inference faithfulness invariant (native encode == trainer encode)."""
import numpy as np
import pytest

import simlens
from simlens import train
from simlens.train import build_bundle, import_sae
from simlens.train.metrics import fvu, l0, rank_preservation, sae_quality


@pytest.fixture(scope="module")
def corpus():
    rng = np.random.default_rng(0)
    dim, kc, n = 48, 6, 1200
    protos = rng.standard_normal((kc, dim))
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    mem = rng.random((n, kc)) < 0.25
    X = (mem @ protos + 0.05 * rng.standard_normal((n, dim))).astype(np.float32)
    return X


@pytest.mark.parametrize("arch,kw", [
    ("topk", {"k": 16}),
    ("batchtopk", {"k": 16}),
    ("jumprelu", {"sparsity": 3e-3}),
    ("relu", {}),
])
def test_trainer_reconstructs(corpus, arch, kw):
    sae = train.fit(corpus, arch=arch, expansion=8, epochs=30, **kw)
    q = sae.quality(corpus)
    assert q["fvu"] < 0.15  # good reconstruction on a well-structured corpus
    assert q["dead_pct"] < 5.0


def test_topk_l0_equals_k(corpus):
    sae = train.fit(corpus, arch="topk", k=8, expansion=8, epochs=20)
    H = sae.encode(corpus)
    assert l0(H) == pytest.approx(8.0, abs=1e-6)  # k directly sets L0


@pytest.mark.parametrize("arch,kw", [
    ("topk", {"k": 12}),
    ("batchtopk", {"k": 12}),
    ("jumprelu", {"sparsity": 3e-3}),
])
def test_native_encode_matches_trainer(corpus, arch, kw):
    """The crux of faithfulness: the Rust kernel must reproduce the trainer's activations,
    gate and all, so attribution uses exactly the codes the trainer optimized."""
    sae = train.fit(corpus, arch=arch, expansion=6, epochs=20, **kw)
    b = build_bundle("syn", "cosine", sae, corpus)
    nat = b.to_native_sae()
    py = sae.encode(corpus[:8])
    for i in range(8):
        ne = np.asarray(nat.encode(corpus[i].tolist()))
        assert np.abs(ne - py[i]).max() < 1e-5


def test_gate_survives_bundle_roundtrip(corpus, tmp_path):
    sae = train.fit(corpus, arch="topk", k=10, expansion=4, epochs=15)
    b = build_bundle("syn", "cosine", sae, corpus)
    assert b.sae_k == 10
    b.save(tmp_path / "b.simlens")
    r = simlens.Bundle.load(tmp_path / "b.simlens")
    assert r.sae_k == 10
    assert r.verify()
    # jumprelu threshold roundtrips too
    sj = train.fit(corpus, arch="jumprelu", sparsity=3e-3, expansion=4, epochs=15)
    bj = build_bundle("syn", "cosine", sj, corpus)
    assert bj.sae_threshold is not None
    bj.save(tmp_path / "bj.simlens")
    rj = simlens.Bundle.load(tmp_path / "bj.simlens")
    assert rj.sae_threshold is not None and rj.verify()


def test_import_sae_with_gate_roundtrips():
    dim, nf = 4, 8
    rng = np.random.default_rng(1)
    w_enc = rng.standard_normal((nf, dim)).astype(np.float32)
    b_enc = np.zeros(nf, np.float32)
    w_dec = rng.standard_normal((dim, nf)).astype(np.float32)
    b = import_sae("ext", "cosine", w_enc, b_enc, w_dec, k=3)
    assert b.sae_k == 3
    nat = b.to_native_sae()
    a = np.asarray(nat.encode(rng.standard_normal(dim).astype(np.float32).tolist()))
    assert (a != 0).sum() <= 3  # top-k gate honored at inference


def test_metrics_edge_cases():
    X = np.zeros((10, 4))
    assert fvu(X, X) == 0.0
    assert l0(np.zeros((3, 5))) == 0.0
    # rank preservation is 1.0 when recon == raw
    R = np.random.default_rng(0).standard_normal((40, 6))
    assert rank_preservation(R, R, k=5) == pytest.approx(1.0)
