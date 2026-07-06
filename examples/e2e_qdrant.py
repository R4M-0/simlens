"""End-to-end integration test: real embedder (BGE via fastembed) + real vector DB
(Qdrant, embedded) + SimLens explanations, with a quantitative quality check.

Requires: pip install fastembed qdrant-client
Run:      python examples/e2e_qdrant.py
"""
from __future__ import annotations

import numpy as np
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import simlens
from simlens import train

TOPICS = {
    "finance": {
        "subj": ["the central bank", "investors", "the hedge fund", "the treasury", "the stock market", "the analyst"],
        "verb": ["raised interest rates on", "sold off", "hedged against inflation in", "reported quarterly earnings for", "shorted", "invested heavily in"],
        "obj": ["government bonds", "the equity portfolio", "emerging markets", "the currency reserves", "corporate debt", "the index fund"],
    },
    "sports": {
        "subj": ["the striker", "the coach", "the goalkeeper", "the marathon runner", "the tennis champion", "the home team"],
        "verb": ["scored a goal against", "defeated", "trained hard for", "won the match versus", "sprinted past", "celebrated victory over"],
        "obj": ["the rival club", "the national squad", "the championship final", "the defending champions", "the away side", "the tournament favorites"],
    },
    "medicine": {
        "subj": ["the physician", "the clinical trial", "the vaccine", "the surgeon", "the patient", "the research team"],
        "verb": ["diagnosed", "reduced the symptoms of", "administered a treatment for", "studied the immune response to", "prescribed antibiotics for", "monitored the recovery from"],
        "obj": ["the chronic infection", "the tumor", "the viral disease", "the inflammation", "the cardiac condition", "the bacterial pathogen"],
    },
    "cooking": {
        "subj": ["the chef", "the recipe", "the baker", "the home cook", "the pastry", "the sauce"],
        "verb": ["simmered", "seasoned", "baked", "caramelized", "whisked together", "roasted"],
        "obj": ["the garlic and onions", "the fresh herbs", "the sourdough loaf", "the tomato broth", "the chocolate batter", "the marinated vegetables"],
    },
    "technology": {
        "subj": ["the engineer", "the startup", "the algorithm", "the data center", "the software team", "the neural network"],
        "verb": ["deployed", "optimized", "trained", "scaled up", "refactored", "benchmarked"],
        "obj": ["the machine learning model", "the cloud infrastructure", "the distributed database", "the compiler backend", "the recommendation system", "the GPU cluster"],
    },
}


def make_corpus(per_topic=40, seed=0):
    rng = np.random.default_rng(seed)
    texts, labels = [], []
    for topic, g in TOPICS.items():
        seen = set()
        while sum(1 for l in labels if l == topic) < per_topic:
            s = f"{rng.choice(g['subj'])} {rng.choice(g['verb'])} {rng.choice(g['obj'])}."
            if s in seen:
                continue
            seen.add(s)
            texts.append(s)
            labels.append(topic)
    return texts, labels


def main():
    topics = list(TOPICS)
    texts, labels = make_corpus()
    print(f"corpus: {len(texts)} docs across {len(topics)} topics")

    print("embedding with BAAI/bge-small-en-v1.5 ...")
    embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
    X = np.asarray(list(embedder.embed(texts)), dtype=np.float32)
    dim = X.shape[1]

    # ---- real vector DB: Qdrant (embedded) -----------------------------------
    client = QdrantClient(location=":memory:")
    client.create_collection("docs", vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
    client.upsert(
        "docs",
        [PointStruct(id=i, vector=X[i].tolist(), payload={"text": texts[i], "topic": labels[i]})
         for i in range(len(texts))],
    )
    print(f"upserted {client.count('docs').count} points into Qdrant")

    # ---- build a SimLens bundle on the real embedding space -------------------
    print("training SAE + building bundle ...")
    sae = train.SAE(dim=dim, expansion=6, l1=2e-3, seed=0).fit(X, epochs=80)
    labelers = {t: np.array([1.0 if l == t else 0.0 for l in labels]) for t in topics}
    bundle = train.build_bundle("bge-small-en-v1.5", "cosine", sae, X, labelers=labelers, items=texts)
    for t in topics:
        pos = X[[i for i, l in enumerate(labels) if l == t]]
        neg = X[[i for i, l in enumerate(labels) if l != t]]
        bundle.add_concept(t, pos, neg, aspect="topic")
    named = sum(1 for n in bundle.feature_names if n)
    print(f"  {bundle.n_features} SAE features, {named} auto-named; "
          f"concept accuracies={ {t: round(c,2) for t,c in zip(bundle.concept_names, bundle.concept_conf)} }")

    ex = simlens.Explainer(bundle)
    store = simlens.adapters.Qdrant("docs", client=client)

    # ---- held-out queries: retrieval + explanation quality --------------------
    queries = [
        ("finance", "The federal reserve tightened monetary policy to curb rising inflation."),
        ("sports", "The forward netted a last-minute winner in the cup final."),
        ("medicine", "Doctors treated the patient's severe lung infection with new drugs."),
        ("cooking", "She slowly braised the beef with rosemary and roasted garlic."),
        ("technology", "The team deployed a large language model to their kubernetes cluster."),
    ]
    qvecs = np.asarray(list(embedder.embed([q for _, q in queries])), dtype=np.float32)

    retrieval_hits = 0
    concept_hits = 0
    adapter_ok = True
    print("\n--- per-query results ---")
    for (true_topic, qtext), qv in zip(queries, qvecs):
        res = client.query_points("docs", query=qv.tolist(), limit=5, with_payload=True).points
        top = res[0]
        retrieved_topic = top.payload["topic"]
        retrieval_hits += retrieved_topic == true_topic

        # fetch the hit's vector back through the SimLens Qdrant adapter (real integration)
        cvec = store.get([top.id])[0]
        if not np.allclose(cvec, X[top.id], atol=1e-5):
            adapter_ok = False

        cattr = ex.explain(qv, cvec, level="concept", top_k=3)
        top_concept = cattr.contributions[0].name if cattr.contributions else None
        concept_hits += top_concept == true_topic

        print(f"[{true_topic:10}] retrieved={retrieved_topic:10} "
              f"top_concept={top_concept:10} resid={cattr.completeness_residual:.3f}")
        print(f"             hit: {top.payload['text']}")
        print(f"             why: {cattr.as_sentence()}")

    n = len(queries)
    print("\n--- quality summary ---")
    print(f"retrieval top-1 accuracy : {retrieval_hits}/{n}")
    print(f"concept-attribution match: {concept_hits}/{n}  (top named concept == true topic)")
    print(f"SimLens Qdrant adapter    : {'OK' if adapter_ok else 'MISMATCH'}")

    # feature-naming sanity: do named features line up with topics?
    from collections import Counter
    name_counts = Counter(n for n in bundle.feature_names if n)
    print(f"named-feature topic spread: {dict(name_counts)}")

    # faithfulness across levels
    pairs = [(X[i], X[i + 1]) for i in range(0, 40, 4)]
    card = simlens.eval.scorecard(ex, pairs)
    for lvl in ("dim", "feature", "concept"):
        r = card[lvl]
        if r:
            print(f"faithfulness {lvl:8}: residual_mean={r['residual_mean']:.4f} "
                  f"rel={r['relative_residual_mean']:.3f}")


if __name__ == "__main__":
    main()
