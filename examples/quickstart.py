"""End-to-end SimLens demo on a synthetic embedding space with known latent concepts.

Run:  python examples/quickstart.py
"""
import tempfile
from pathlib import Path

import numpy as np

import simlens
from simlens import train


def make_space(dim=48, n_concepts=6, n=500, seed=0):
    """Vectors built as sparse combinations of latent concept prototypes + noise."""
    rng = np.random.default_rng(seed)
    protos = rng.standard_normal((n_concepts, dim))
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    membership = rng.random((n, n_concepts)) < 0.3
    X = membership.astype(float) @ protos + 0.05 * rng.standard_normal((n, dim))
    return X.astype(np.float32), membership, protos


def main():
    X, membership, _ = make_space()
    dim, n_concepts = X.shape[1], membership.shape[1]
    names = [f"concept_{k}" for k in range(n_concepts)]

    print("== 1. zero-setup Level-1 (no training) ==")
    ex0 = simlens.Explainer(metric="cosine")
    a, b = X[0], X[1]
    print("   ", ex0.explain(a, b).as_sentence())

    print("\n== 2. train a sparse autoencoder + auto-label features ==")
    sae = train.SAE(dim=dim, expansion=4, l1=1e-3, seed=0).fit(X, epochs=60, verbose=False)
    labelers = {names[k]: membership[:, k].astype(float) for k in range(n_concepts)}
    bundle = train.build_bundle("synthetic-v1", "cosine", sae, X, labelers=labelers)
    named = [n for n in bundle.feature_names if n]
    print(f"    {bundle.n_features} features, {len(named)} auto-named "
          f"(e.g. {sorted(set(named))[:4]})")

    print("\n== 3. register named concepts (CAVs) ==")
    for k in range(n_concepts):
        bundle.add_concept(
            names[k], X[membership[:, k]], X[~membership[:, k]], aspect="topic"
        )
    print("    concept accuracies:",
          {n: round(c, 2) for n, c in zip(bundle.concept_names, bundle.concept_conf)})

    print("\n== 4. save / load / verify (audit) ==")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "demo.simlens"
        bundle.save(path)
        reloaded = simlens.Bundle.load(path)
        print(f"    hash {reloaded.content_hash[:23]}…  verify={reloaded.verify()}")
        ex = simlens.Explainer(reloaded)

        # two items that both carry concept_2
        idx = np.where(membership[:, 2])[0]
        q, c = X[idx[0]], X[idx[1]]

        print("\n== 5. feature-level explanation ==")
        print(simlens.viz.render(ex.explain(q, c, level="feature", top_k=5)))

        print("\n== 6. concept-level explanation ==")
        print(simlens.viz.render(ex.explain(q, c, level="concept", top_k=4)))

        print("\n== 7. aspect view ==")
        print("   ", ex.explain(q, c, level="aspect").as_sentence())

        print("\n== 8. why does 'better' outrank 'worse'? (margin) ==")
        worse = X[np.where(~membership[:, 2])[0][0]]
        m = ex.explain_margin(q, c, worse, level="concept", top_k=3)
        print("   ", m.as_sentence())

        print("\n== 9. ablation: minimal features to break the match ==")
        abl = ex.ablate(q, c, threshold=0.5)
        print(f"    removing {len(abl['removed'])} features: "
              f"{abl['score_before']:.3f} → {abl['score_after']:.3f} "
              f"(dropped_below={abl['dropped_below']})")

        print("\n== 10. faithfulness scorecard ==")
        pairs = [(X[i], X[i + 1]) for i in range(0, 20, 2)]
        card = simlens.eval.scorecard(ex, pairs)
        for lvl in ("dim", "feature", "concept"):
            r = card[lvl]
            if r:
                print(f"    {lvl:8} residual_mean={r['residual_mean']:.4f} "
                      f"rel={r['relative_residual_mean']:.3f} coverage={r['coverage_mean']:.0%}")

        print("\n== 11. why NOT more similar? (dissimilarity) ==")
        c_other = X[np.where(~membership[:, 2])[0][0]]
        dis = ex.explain_dissimilarity(q, c_other, top_k=4)
        print("   ", ", ".join(f"{d.label}[{d.polarity}]" for d in dis.contributions))

        print("\n== 12. confidence calibration ==")
        rep = simlens.eval.reliability(reloaded, X, labelers)
        print(f"    {rep['n_named']} named features, "
              f"calibration_error={rep['calibration_error']}")

    print("\n== 13. late-interaction (multi-vector) attribution ==")
    rng = np.random.default_rng(7)
    Cq = rng.standard_normal((5, 32))
    Cc = np.vstack([Cq[1], rng.standard_normal((4, 32))])  # candidate shares one token
    mv = simlens.MultiVectorExplainer("cosine").explain(
        Cq, Cc, query_labels=[f"q{i}" for i in range(5)], candidate_labels=[f"c{j}" for j in range(5)]
    )
    print(f"    score={mv.score:.3f} residual={mv.completeness_residual:.1e}")
    print("   ", mv.as_sentence())


if __name__ == "__main__":
    main()
