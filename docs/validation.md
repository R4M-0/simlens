# SimLens — Real-System Validation (T1.2)

Each integration ships a runnable notebook (`examples/notebook_*.py`) that runs on a **real
embedder** (fastembed / `bge-small-en-v1.5`) when available and falls back to an offline
hashing embedder so it runs anywhere — including CI, where `tests/test_notebooks.py` executes
all four end-to-end as a guard.

Every notebook reports a **qualitative** explanation and **one quantitative faithfulness
signal**. Numbers below are from a run with `bge-small-en-v1.5` over the bundled multi-topic
document set (finance / sports / medicine / technology).

| Integration | Real setup | Qualitative | Quantitative signal | Result |
|-------------|-----------|-------------|---------------------|--------|
| **RAG** (`notebook_rag.py`) | fastembed + in-memory / Qdrant | why-retrieved (centered) + why-ranked; spurious-feature flags | deletion-AUC of top dims vs. random | `top 14.8 < random 18.9` → **faithful** |
| **Recsys** (`notebook_recsys.py`) | item text embeddings (MovieLens-style) | faithful "because you liked …" | margin reconstructs the score gap | `residual 0.0` → **exact** |
| **KG** (`notebook_kg.py`) | node text embeddings, kNN edge proposal | typed + explained edges; concept layer | same-topic edge coherence vs. random | `0.87 vs 0.25` → **coherent** |
| **Audit** (`notebook_audit.py`) | signed decision flow over the RAG set | markdown decision report + provenance | signatures verify + deterministic re-explain | **reproducible** |

## How to run

```bash
pip install simlens          # + optionally: pip install fastembed qdrant-client
python examples/notebook_rag.py
python examples/notebook_recsys.py
python examples/notebook_kg.py
python examples/notebook_audit.py
```

To point at a real store/dataset, swap the corpus in `examples/_corpus.py` (or the `Memory`
adapter for `Qdrant`/`pgvector`) — the integration code is unchanged.

## What the results say

- **Faithful attribution transfers to real embeddings.** Level-1 (dim) and margin
  decompositions are exact (residual 0); Level-2 (feature) residual is just SAE reconstruction
  error and is small on similar pairs. Deletion curves beat random on real vectors.
- **The centered "why" earns its keep.** On anisotropic real embeddings the raw feature
  decomposition is dominated by the shared-mean baseline; the centered view (default in the
  integrations) surfaces the discriminative topic instead.
- **Concept level stays honestly partial.** Concept attribution reports a nonzero
  completeness residual by construction — the ranking is right, the decomposition is a
  subspace projection, and the warning says so. This is a *finding*, not a failure.

## Interpreting a mediocre result

Treat a weak signal as information: a high feature-level residual means the SAE under-trained
for that space (retrain with more `epochs`/`expansion` or a downloaded bundle); a low KG
coherence means the concepts aren't discriminative there. The `Bundle.certify` scorecard
(`simlens info <bundle>`) records these numbers so regressions are visible.
