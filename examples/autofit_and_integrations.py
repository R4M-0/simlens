"""Part A (autofit) + Part B (integrations) demo — no external services or models.

Shows the low-burden path: point SimLens at a store with payloads, get a named,
concept-annotated bundle automatically, then use the system extensions.

Run:  python examples/autofit_and_integrations.py
"""
import numpy as np

import simlens
from simlens.adapters import Memory
from simlens.integrations import (
    AuditLog,
    KnowledgeGraphExplainer,
    RagExplainer,
    RecsysExplainer,
)


def build_store(n=210, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    topics = ["finance", "sports", "medicine"]
    protos = rng.standard_normal((len(topics), dim))
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    store, X, labels = Memory(), [], []
    for i in range(n):
        t = i % len(topics)
        v = (protos[t] + 0.07 * rng.standard_normal(dim)).astype(np.float32)
        store.add(f"id{i}", v, payload={"topic": topics[t], "approved": t == 0,
                                        "text": f"{topics[t]} document number {i}"})
        X.append(v)
        labels.append(t)
    return store, np.asarray(X, np.float32), labels


def main():
    store, X, labels = build_store()

    print("== Part A: autofit (no manual corpus / labels / concepts) ==")
    bundle = simlens.autofit(store, embedder="demo", metric="cosine", expansion=4, epochs=40)
    named = sum(1 for x in bundle.feature_names if x)
    print(f"   features named: {named}/{bundle.n_features}  provenance={bundle.name_provenance()}")
    print(f"   auto-discovered concepts: {bundle.concept_names}")
    print("   (LLM naming: pass namer=simlens.naming.from_provider('openai'|'gemini'))")

    fin = [i for i, l in enumerate(labels) if l == 0]

    print("\n== Part B — knowledge graph ==")
    kg = KnowledgeGraphExplainer(bundle)
    print("   finance–finance edge:", kg.explain_edge(X[fin[0]], X[fin[1]])["explanation"])
    nodes = {f"n{i}": X[i] for i in range(9)}
    print(f"   proposed subgraph edges: {len(kg.propose_edges(nodes, threshold=0.5))}")
    print(f"   concept-layer edges:     {len(kg.concept_layer(nodes))}")

    print("\n== Part B — RAG / search ==")
    rag = RagExplainer(bundle, store=store, spurious={"topic=sports"})
    row = rag.explain_results(X[fin[0]], ["id3", "id6", "id0"])[0]
    print("   top hit why:", row["why"])

    print("\n== Part B — recommender ==")
    rec = RecsysExplainer(bundle)
    print("   because you liked …:", rec.because([X[i] for i in fin[:3]], X[fin[0]])["why"])

    print("\n== Part B — audit (regulated) ==")
    al = AuditLog(bundle, secret="org-secret")
    rec_ = al.record(X[fin[0]], X[fin[1]], decision="link_created", context={"campaign": "demo"})
    print(f"   signed record verified: {al.verify(rec_)}  provenance={rec_['bundle_hash'][:20]}…")


if __name__ == "__main__":
    main()
