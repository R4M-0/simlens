"""T1.2 — Recommender validation on a real system.

Real setup: MovieLens items embedded from their text (title + genres/tags). Offline, the same
topical corpus stands in for "items", each topic acting as a taste cluster.

Output: (qualitative) a faithful "because you liked …" reason; (quantitative) margin
faithfulness — the explained margin between a good and a worse recommendation reconstructs
the actual score gap.

    python examples/notebook_recsys.py
"""
from __future__ import annotations

import numpy as np

import simlens
from simlens.integrations import RecsysExplainer

from _corpus import topic_corpus


def main():
    vecs, texts, payloads, labels, topics, backend = topic_corpus(repeat=8)
    print(f"[recsys] item embedder: {backend}  |  {len(texts)} items")

    bundle = simlens.autofit(vectors=vecs, payloads=payloads, embedder=backend,
                             metric="cosine", expansion=6, epochs=25)
    rec = RecsysExplainer(bundle)

    # a user who likes "technology" items
    tech = np.where(labels == topics.index("technology"))[0]
    liked = vecs[tech[:3]]
    # a good candidate (another tech item) and a worse one (finance)
    good = vecs[tech[3]]
    worse = vecs[np.where(labels == topics.index("finance"))[0][0]]

    out = rec.because(liked, good)
    print(f"\n[recsys] because you liked …: {out['why']}")
    for c in out["concepts"][:3]:
        print(f"    {c['name']}  weight={c['weight']}  evidence={ (c['evidence'] or [''])[0][:48] }")

    # quantitative: margin faithfulness — Σφ reconstructs the score gap
    ex = simlens.Explainer(bundle)
    m = ex.explain_margin(np.mean(liked, axis=0), good, worse, level="feature", top_k=10_000)
    gap = float(ex.explain(np.mean(liked, axis=0), good, level="feature").score
                - ex.explain(np.mean(liked, axis=0), worse, level="feature").score)
    residual = abs(m.score - gap)
    print(f"\n[recsys] FAITHFULNESS margin={m.score:.4f} vs score-gap={gap:.4f} "
          f"residual={residual:.2e}  faithful={residual < 1e-6}")
    return residual < 1e-6


if __name__ == "__main__":
    main()
