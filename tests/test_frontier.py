"""Frontier items: cross-modal concepts and hierarchical (meta) concepts."""
import numpy as np
import pytest

import simlens
from simlens import train
from simlens.train import build_bundle


def test_cross_modal_concept_is_tagged_and_usable():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 16)).astype(np.float32)
    sae = train.fit(X, arch="topk", k=4, expansion=3, epochs=8)
    b = build_bundle("clip-img", "cosine", sae, X, modality="image")
    text_dir = rng.standard_normal(16).astype(np.float32)  # a "text" concept direction
    b.add_cross_modal_concept("a photo of a cat", text_dir, source_modality="text")
    assert "a photo of a cat" in b.concept_names
    k = b.concept_names.index("a photo of a cat")
    assert b.concept_source[k] == "cross_modal"
    assert b.aspects["a photo of a cat"] == "cross-modal:text→image"
    # it projects like any concept
    ex = simlens.Explainer(b)
    a = ex.explain(X[0], X[1], level="concept")
    assert any(c.name == "a photo of a cat" for c in a.contributions)


def test_hierarchy_rolls_cofiring_features_into_meta_concepts():
    # build a corpus with clear feature groups: two blocks of dims always co-activate
    rng = np.random.default_rng(1)
    dim, n = 20, 800
    X = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        if rng.random() < 0.5:
            X[i, 0:5] = rng.random(5) + 1.0   # group A co-fires
        if rng.random() < 0.5:
            X[i, 10:15] = rng.random(5) + 1.0  # group B co-fires
    X += 0.02 * rng.standard_normal((n, dim)).astype(np.float32)
    sae = train.fit(X, arch="topk", k=8, expansion=4, epochs=25)
    b = build_bundle("syn", "cosine", sae, X)
    before = len(b.concept_names)
    metas = simlens.discover_hierarchy(b, X, threshold=0.3, min_size=2)
    assert metas  # at least one meta-concept discovered
    assert len(b.concept_names) == before + len(metas)
    for m in metas:
        assert len(m["feature_indices"]) >= 2
        k = b.concept_names.index(m["name"])
        assert b.concept_source[k] == "hierarchy"
        assert b.aspects[m["name"]] == "meta"


def test_save_rust_emits_loadable_form(tmp_path):
    pytest.importorskip("safetensors")
    rng = np.random.default_rng(3)
    X = rng.standard_normal((60, 8)).astype(np.float32)
    sae = train.fit(X, arch="topk", k=3, expansion=3, epochs=8)
    b = build_bundle("rust", "cosine", sae, X)
    b.save_rust(tmp_path / "b.simlens")
    # the Rust-loadable artifacts exist and safetensors round-trips the weights
    assert (tmp_path / "b.simlens" / "manifest.json").exists()
    from safetensors.numpy import load_file

    tensors = load_file(str(tmp_path / "b.simlens" / "weights.safetensors"))
    assert tensors["w_enc"].shape == (sae.n_features, sae.dim)
    import json as _json

    man = _json.loads((tmp_path / "b.simlens" / "manifest.json").read_text())
    assert man["sae"]["k"] == 3


def test_hierarchy_requires_sae():
    b = simlens.Bundle(embedder="e", dim=4)
    try:
        simlens.discover_hierarchy(b, np.ones((5, 4)))
        assert False, "expected ValueError"
    except ValueError:
        pass
