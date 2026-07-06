import numpy as np
import pytest

import simlens
from simlens.adapters import Memory
from simlens.integrations import (
    AuditLog,
    KnowledgeGraphExplainer,
    RagExplainer,
    RecsysExplainer,
)


@pytest.fixture(scope="module")
def store_and_labels():
    rng = np.random.default_rng(0)
    dim, k, n = 24, 3, 180
    protos = rng.standard_normal((k, dim))
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    topics = ["finance", "sports", "medicine"]
    store = Memory()
    X, lab = [], []
    for i in range(n):
        t = i % k
        v = (protos[t] + 0.06 * rng.standard_normal(dim)).astype(np.float32)
        store.add(f"id{i}", v, payload={"topic": topics[t], "approved": t == 0,
                                        "text": f"{topics[t]} document {i}"})
        X.append(v)
        lab.append(t)
    return store, np.asarray(X, np.float32), lab, topics


@pytest.fixture(scope="module")
def bundle(store_and_labels):
    store, X, lab, topics = store_and_labels
    return simlens.autofit(store, embedder="syn", metric="cosine", expansion=4, epochs=30)


# ---- Part A: autofit ----------------------------------------------------------
def test_autofit_names_features_from_payload(bundle):
    named = sum(1 for x in bundle.feature_names if x)
    assert named > 0
    prov = bundle.name_provenance()
    assert prov.get("payload", 0) > 0  # payload fields auto-named features


def test_autofit_discovers_concepts_from_payload(bundle):
    # topic=* and approved should be auto-registered as concepts
    assert any(n.startswith("topic=") for n in bundle.concept_names)
    assert "approved" in bundle.concept_names


def test_derive_payload_labelers():
    from simlens.autofit import derive_payload_labelers

    payloads = [{"topic": "a", "n": 1.0, "ok": True}, {"topic": "b", "n": 2.0, "ok": False}]
    labs = derive_payload_labelers(payloads)
    assert "topic=a" in labs and "n" in labs and "ok" in labs


def test_text_concepts_via_embed_callable(store_and_labels):
    store, X, lab, topics = store_and_labels
    # a trivial "embedder" mapping words to random-but-fixed vectors in the same space
    rng = np.random.default_rng(1)
    vocab = {w: rng.standard_normal(X.shape[1]).astype(np.float32) for w in ["risky", "safe"]}
    b = simlens.autofit(
        store, embedder="syn", metric="cosine", expansion=4, epochs=20,
        text_concepts={"risky": "risky", "safe": "safe"},
        embed_text=lambda s: vocab[s],
    )
    assert "risky" in b.concept_names
    k = b.concept_names.index("risky")
    assert b.concept_source[k] == "text"


# ---- naming -------------------------------------------------------------------
def test_keyword_namer():
    from simlens.naming import KeywordNamer

    name, conf = KeywordNamer().name(["finance report quarterly", "finance market crash"])
    assert name and "finance" in name and 0 <= conf <= 1


def test_llm_namer_from_callable_tags_ai():
    from simlens.naming import from_callable

    namer = from_callable(lambda prompt: "kinase inhibitor")
    assert namer.source == "ai"
    name, conf = namer.name(["mol A", "mol B"])
    assert name == "kinase inhibitor"


def test_rename_feature_override(bundle):
    b = simlens.Bundle(**{**bundle.__dict__})  # shallow copy is fine for this check
    i = next(i for i, n in enumerate(b.feature_names) if n)
    b.rename_feature(i, "my label")
    assert b.feature_names[i] == "my label"
    assert b.feature_source[i] == "manual"


# ---- Part B: KG ---------------------------------------------------------------
def test_kg_explain_edge_and_propose(bundle, store_and_labels):
    _, X, lab, _ = store_and_labels
    kg = KnowledgeGraphExplainer(bundle)
    fin = [i for i in range(len(lab)) if lab[i] == 0][:2]
    edge = kg.explain_edge(X[fin[0]], X[fin[1]])
    assert edge["explanation"].startswith("Similar")
    assert "contributingDimensions" in edge and edge["completenessResidual"] >= 0
    nodes = {f"n{i}": X[i] for i in range(6)}
    edges = kg.propose_edges(nodes, threshold=0.4)
    assert all("type" in e and "source" in e for e in edges)


def test_kg_concept_layer(bundle, store_and_labels):
    _, X, _, _ = store_and_labels
    kg = KnowledgeGraphExplainer(bundle)
    layer = kg.concept_layer({f"n{i}": X[i] for i in range(4)}, top_k=2)
    assert all(e["target"].startswith("concept:") for e in layer)


# ---- Part B: RAG / Recsys / Audit --------------------------------------------
def test_rag_results_and_spurious_flag(bundle, store_and_labels):
    store, X, lab, _ = store_and_labels
    rag = RagExplainer(bundle, store=store, spurious={"topic=sports"})
    res = rag.explain_results(X[0], ["id3", "id6", "id0"])
    assert len(res) == 3 and "why" in res[0]
    assert "margin_vs_top" in res[1]


def test_recsys_because_and_steer(bundle, store_and_labels):
    _, X, lab, _ = store_and_labels
    rec = RecsysExplainer(bundle)
    fin = [i for i in range(len(lab)) if lab[i] == 0][:2]
    out = rec.because([X[i] for i in fin], X[fin[0]])
    assert "why" in out
    nudged = rec.nudge(X[0], more=[X[fin[0]]], less=[X[1]])
    assert nudged.shape == X[0].shape


def test_audit_record_sign_verify_report(bundle, store_and_labels):
    _, X, _, _ = store_and_labels
    al = AuditLog(bundle, secret="k")
    r = al.record(X[0], X[3], decision="link", context={"campaign": "c1"})
    assert al.verify(r)
    assert "record_hash" in r
    md = AuditLog.report([r])
    assert "Decision report" in md
