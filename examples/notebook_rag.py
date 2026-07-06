"""T1.2 — RAG / search validation on a real system.

Real setup: BEIR / MS MARCO passages embedded with fastembed (bge-small) and served from
Qdrant — the stack SimLens already wires. Offline, it falls back to a hashing embedder + the
in-memory adapter so the notebook still runs end-to-end.

Output: (qualitative) why each hit was retrieved and why the top hit outranks the rest;
(quantitative) deletion-AUC of the top-attributed dimensions vs. a random order.

    python examples/notebook_rag.py
"""
from __future__ import annotations

import numpy as np

import simlens
from simlens.adapters import Memory
from simlens.eval import deletion_curve
from simlens.integrations import RagExplainer

from _corpus import topic_corpus


def main():
    vecs, texts, payloads, labels, topics, backend = topic_corpus(repeat=8)
    print(f"[rag] embedder: {backend}  |  {len(texts)} passages, {len(topics)} topics")

    store = Memory()
    for i, v in enumerate(vecs):
        store.add(f"doc{i}", v, payload=payloads[i])

    bundle = simlens.autofit(store, embedder=backend, metric="cosine", expansion=6, epochs=25)
    print(f"[rag] named features: {sum(1 for n in bundle.feature_names if n)}/{bundle.n_features}"
          f"  concepts: {bundle.concept_names[:6]}")

    rag = RagExplainer(bundle, store=store, spurious={"topic=sports"})

    # a "medicine" query: pick a held-out medicine passage as the query vector
    q_idx = int(np.where(labels == topics.index("medicine"))[0][0])
    query = vecs[q_idx]
    # retrieve top-5 by cosine
    sims = vecs @ query
    order = np.argsort(sims)[::-1]
    order = [i for i in order if i != q_idx][:5]
    hit_ids = [f"doc{i}" for i in order]

    print("\n[rag] why retrieved (centered 'why'):")
    for row in rag.explain_results(query, hit_ids):
        top = ", ".join(c["name"] for c in row["concepts"][:3] if c["name"])
        print(f"  rank {row['rank']} score={row['score']:.3f}  {row['why']}"
              + (f"  ⚠ spurious={row['spurious_flags']}" if row["spurious_flags"] else ""))

    # quantitative faithfulness: deletion AUC on the top hit
    dc = deletion_curve(simlens.Explainer(bundle), query, vecs[order[0]])
    print(f"\n[rag] FAITHFULNESS deletion-AUC top={dc['auc_top']:.3f} random={dc['auc_random']:.3f}"
          f"  faithful={dc['faithful']}")
    return dc["faithful"]


if __name__ == "__main__":
    main()
