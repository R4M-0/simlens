"""T1.2 — Audit-trail validation on a realistic decision flow.

A realistic decision flow over the RAG document set: for each query we make a "link this
passage" decision and record a signed, hashed, reproducible audit entry carrying the
explanation and the bundle provenance hash.

Output: (qualitative) a markdown decision report; (quantitative) a reproducibility check —
every signed record verifies, and re-explaining the same pair yields an identical score
(deterministic provenance).

    python examples/notebook_audit.py
"""
from __future__ import annotations

import numpy as np

import simlens
from simlens.integrations import AuditLog

from _corpus import topic_corpus


def main():
    vecs, texts, payloads, labels, topics, backend = topic_corpus(repeat=8)
    print(f"[audit] embedder: {backend}  |  {len(texts)} passages")

    bundle = simlens.autofit(vectors=vecs, payloads=payloads, embedder=backend,
                             metric="cosine", expansion=6, epochs=20)
    bundle.sign("regulator-secret")  # sign the bundle itself
    al = AuditLog(bundle, secret="regulator-secret")

    records = []
    for topic in ("finance", "medicine", "sports"):
        idx = np.where(labels == topics.index(topic))[0]
        q, cand = vecs[idx[0]], vecs[idx[1]]
        rec = al.record(q, cand, decision="link", context={"topic": topic, "policy": "v1"})
        records.append(rec)

    print("\n" + AuditLog.report(records))

    # quantitative reproducibility: signatures verify AND re-explaining reproduces the score
    all_verify = all(al.verify(r) for r in records)
    ex = simlens.Explainer(bundle)
    idx = np.where(labels == topics.index("finance"))[0]
    s1 = ex.explain(vecs[idx[0]], vecs[idx[1]]).score
    s2 = ex.explain(vecs[idx[0]], vecs[idx[1]]).score
    reproducible = abs(s1 - s2) < 1e-12 and bundle.verify_signature("regulator-secret")
    print(f"[audit] FAITHFULNESS all-signatures-verify={all_verify}  "
          f"reproducible={reproducible}  bundle_hash={bundle.content_hash[:20]}…")
    return all_verify and reproducible


if __name__ == "__main__":
    main()
