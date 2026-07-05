<p align="center">
  <img src="assets/logo.svg" alt="SimLens" width="140">
</p>

<h1 align="center">SimLens</h1>

<p align="center">
  <strong>See <em>why</em> your vectors match.</strong><br>
  Faithful, vector-only similarity &amp; ranking attribution — for any embedder, any vector store.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-4f46e5"></a>
  <img alt="Core: Rust" src="https://img.shields.io/badge/core-Rust-4f46e5">
  <img alt="Python 3.8+" src="https://img.shields.io/badge/python-3.8%2B-4f46e5">
  <img alt="Status: v0.1" src="https://img.shields.io/badge/status-v0.1-22c55e">
</p>

---

Vector search powers your RAG pipeline, your recommendations, your semantic search, your
anomaly detection. And every one of them answers a question with a **black box**: *this
matches — score `0.83`.* Which concepts drove the match? Why did the wrong result rank
first? Why was the right one buried at #7? The score won't say.

**SimLens turns that score into an answer** — the specific, named concepts that produced
it, in plain language, and it *proves* the breakdown adds up. It reads only the vectors,
so it drops into **any** stack without touching your model or your database.

> Think of it as **SHAP for vector search**: the missing explanation layer for
> similarity and ranking.

```python
import simlens

# Zero setup — an exact explanation on the very first call, no training required.
ex = simlens.Explainer(metric="cosine")
attr = ex.explain(query_vec, candidate_vec)

print(attr.as_sentence())
# → "Matched mainly on 'financial-regulation' (61%), '2024-filings' (22%)."
```

New here? **[docs/how-simlens-works.md](./docs/how-simlens-works.md)** is a friendly,
plain-language tour of everything below.

## Why teams reach for SimLens

- **Answers the real questions.** Not just *why does A match?* but **why did A outrank
  B?** — the margin, decomposed. The question search and recsys teams actually debug.
- **Faithful by construction.** For dot &amp; cosine, the explanation *is* the arithmetic:
  contributions provably sum to the score. Every result ships a **completeness residual**
  so you can see exactly how exact it is — no hand-wavy storytelling.
- **Three levels of zoom.** Raw dimensions (exact) → learned **monosemantic features** →
  your own **named concepts** — one consistent, additive contract across all three.
- **Steerable.** "More like this, but less of *that*" — edit a query in concept space and
  search again.
- **Drop-in &amp; fast.** Embedder-agnostic, database-agnostic, and backed by a compact
  Rust core for the per-query hot path.
- **Auditable.** Every explanation is stamped with a content hash of the artifacts that
  produced it — reproducible rationale for decisions that have to hold up.

## What you can do

| Capability | Call |
|-----------|------|
| Explain a match (dims / features / concepts / aspects) | `ex.explain(q, c, level=...)` |
| Explain a **ranking** — why A beat B | `ex.explain_margin(q, better, worse)` |
| Minimal reason — what breaks the match | `ex.ablate(q, c, threshold=...)` |
| Steer a query in concept space | `ex.steer(q, {"topic": -1.0})` |
| Contrast against a background set | `ex.explain_vs_corpus(q, c, foil)` |
| Summarize a whole result page | `ex.summarize(q, hits)` |
| Train &amp; package a concept bundle | `simlens.train.build_bundle(...)` |
| Serve over HTTP | `python -m simlens.serve --bundle b.simlens` |

## Install / build from source

```bash
python -m venv .venv && . .venv/bin/activate
pip install maturin numpy pytest
maturin develop --release      # builds the Rust core into your environment
cargo test -p simlens-core     # Rust unit tests
pytest -q                      # Python tests
python examples/quickstart.py  # full end-to-end demo
```

## Project layout

| Path | What |
|------|------|
| `crates/simlens-core` | Rust attribution kernels — the math and the hot path |
| `crates/simlens-py` | PyO3 bindings → `simlens._native` |
| `python/simlens` | Python API: `Explainer`, `Bundle`, `train`, `eval`, `adapters`, `viz`, `serve` |
| `examples/` | Runnable, self-contained demo |
| `docs/` | [How SimLens works](./docs/how-simlens-works.md) |

## Status

**v0.1 — a working, tested MVP.** Level-1/2/3 attribution, margin, ablation, steering,
contrastive-corpus and aspect views, a dependency-light numpy SAE trainer, CAV fitting,
faithfulness evaluation, store adapters, and a serving sidecar — Rust core and Python
suites green. A native Rust `simlens-serve`, zero-copy numpy, and late-interaction
(multi-vector) attribution are on the roadmap.

## Documentation

- **[How SimLens works](./docs/how-simlens-works.md)** — the concepts, end to end.
- **[LICENSE](./LICENSE)** — Apache-2.0.

## License

SimLens is released under the [Apache License 2.0](./LICENSE).
