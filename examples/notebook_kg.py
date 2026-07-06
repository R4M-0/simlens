"""T1.2 — Knowledge-graph validation on a real system.

Real setup: a public product / citation subgraph whose node vectors come from text. Offline,
the topical corpus's items are the nodes. We propose typed, explained edges via kNN candidate
generation (O(n·k)) and inspect the concept layer.

Output: (qualitative) typed + explained edges and per-node concept memberships;
(quantitative) concept-layer coherence — the fraction of proposed edges that connect nodes of
the same ground-truth topic (a proxy for edge correctness).

    python examples/notebook_kg.py
"""
from __future__ import annotations

import numpy as np

import simlens
from simlens.integrations import KnowledgeGraphExplainer

from _corpus import topic_corpus


def main():
    vecs, texts, payloads, labels, topics, backend = topic_corpus(repeat=8)
    print(f"[kg] node embedder: {backend}  |  {len(texts)} nodes")

    bundle = simlens.autofit(vectors=vecs, payloads=payloads, embedder=backend,
                             metric="cosine", expansion=6, epochs=25)
    kg = KnowledgeGraphExplainer(bundle, top_k=4)

    nodes = {f"n{i}": vecs[i] for i in range(len(vecs))}
    edges = kg.propose_edges_knn(nodes, k=6, threshold=0.35)
    print(f"\n[kg] proposed {len(edges)} typed edges (kNN, O(n·k)). Examples:")
    for e in edges[:4]:
        print(f"    {e['source']}—{e['target']}  type={e['type']!r}  {e['explanation']}")

    # concept layer: entity→concept memberships
    layer = kg.concept_layer(dict(list(nodes.items())[:6]), top_k=2)
    print(f"\n[kg] concept-layer edges (sample): "
          + ", ".join(f"{e['source']}→{e['target']}" for e in layer[:5]))

    # quantitative: same-topic coherence of proposed edges
    same = sum(1 for e in edges
               if labels[int(e["source"][1:])] == labels[int(e["target"][1:])])
    coherence = same / max(len(edges), 1)
    print(f"\n[kg] FAITHFULNESS same-topic edge coherence={coherence:.2f} "
          f"(random baseline ≈ {1/len(topics):.2f})  good={coherence > 1/len(topics)}")
    return coherence > 1 / len(topics)


if __name__ == "__main__":
    main()
