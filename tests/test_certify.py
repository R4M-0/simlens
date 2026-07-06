"""T2.3 — faithfulness certification stored (and hash-covered) in the bundle."""
import json

import numpy as np
import pytest

import simlens


@pytest.fixture(scope="module")
def corpus():
    rng = np.random.default_rng(0)
    dim, kc, n = 32, 5, 400
    protos = rng.standard_normal((kc, dim))
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    mem = rng.random((n, kc)) < 0.3
    X = (mem @ protos + 0.05 * rng.standard_normal((n, dim))).astype(np.float32)
    return X


def test_certify_populates_scorecard(corpus):
    b = simlens.autofit(vectors=corpus, embedder="syn", metric="cosine", expansion=4,
                        epochs=15, certify=False)
    card = b.certify(corpus, n_pairs=15)
    assert "sae" in card and 0.0 <= card["sae"]["fvu"]
    assert card["sae"]["l0"] > 0
    assert "attribution" in card and card["attribution"]["dim"]["residual_max"] < 1e-3
    assert card["attribution"]["deletion_insertion"]["deletion_faithful"]
    assert "concepts" in card


def test_autofit_certifies_by_default(corpus):
    b = simlens.autofit(vectors=corpus, embedder="syn", metric="cosine", expansion=4, epochs=10)
    assert b.faithfulness  # non-empty
    assert "sae" in b.faithfulness


def test_faithfulness_survives_roundtrip_and_hash(corpus, tmp_path):
    b = simlens.autofit(vectors=corpus, embedder="syn", metric="cosine", expansion=4, epochs=10)
    assert b.verify()  # hash covers the scorecard
    b.save(tmp_path / "b.simlens")
    r = simlens.Bundle.load(tmp_path / "b.simlens")
    assert r.faithfulness == b.faithfulness
    assert r.verify()


def test_tampering_with_scorecard_breaks_hash(corpus):
    b = simlens.autofit(vectors=corpus, embedder="syn", metric="cosine", expansion=4, epochs=10)
    assert b.verify()
    b.faithfulness["sae"]["fvu"] = 0.0  # forge a better number
    assert not b.verify()  # hash no longer matches


def test_manifest_exposes_faithfulness(corpus):
    b = simlens.autofit(vectors=corpus, embedder="syn", metric="cosine", expansion=4, epochs=10)
    man = json.loads(json.dumps(b._manifest()))  # serializable
    assert man["faithfulness"] is not None
    assert man["sae"]["gate"] == "topk"
