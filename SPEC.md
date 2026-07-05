# SimLens — Specification

> **SimLens** is a similarity-attribution engine — a *lens* you look through to see why a
> similarity holds. It takes a similarity or ranking score produced by any embedding
> model and **resolves** it into an additive, axiom-satisfying spectrum of **named
> concept contributions** — telling you not just *that* two vectors are similar, but
> *why*, in human terms.
>
> Package/crate/npm name `simlens` — verified available on PyPI, crates.io, and npm
> (July 2026).

**Status:** Draft v0.1 — design spec, pre-implementation.
**License (intended):** Apache-2.0.

---

## 0. One-paragraph summary

Vector databases, RAG pipelines, recommender systems, and semantic search all reduce a
rich comparison between two embeddings to a single number (a cosine score, a dot
product, a distance). That number is opaque: it says *A is close to B* but never *why*.
SimLens is a library and sidecar that **decomposes any similarity or ranking score into
additive contributions** at three zoom levels — raw **dimensions**, learned
**monosemantic features**, and human-named **concepts** — with a mathematical
**completeness guarantee** (the contributions provably sum back to the score). It is
**embedder-agnostic** and **database-agnostic**: it needs only the vectors, which every
vector store can already return. A fast Rust core handles the per-query hot path; a
Python package handles offline training of the concept dictionaries; a thin serving
binary exposes it to any language. Think of it as **SHAP for vector search**.

---

## 1. The problem

### 1.1 The gap in one sentence

The *science* of explaining embedding similarity exists — but it is fragmented across
research papers, is mostly **text-only** and **offline**, and each method delivers **one**
kind of explanation. There is no **unified, production-grade, embedder- and
database-agnostic** library that explains **why two embeddings are similar** or **why one
result outranked another** — across zoom levels, with a completeness guarantee, at
serving time, in a form a human can read and audit.

> **Honesty note.** SimLens is *consolidation + productization + faithfulness guarantees
> + generality*, not a first mover on the underlying ideas. A full prior-art map and a
> critical self-evaluation live in [`EVALUATION.md`](./EVALUATION.md). Read it alongside
> this spec — several design decisions below exist specifically to differentiate from,
> or interoperate with, existing work.

### 1.2 Why existing tools do not cover it

| Family | What it explains | Why it is not this |
|--------|------------------|--------------------|
| SHAP, LIME, Captum, Integrated Gradients | a model's **prediction** from its **input features** | They explain a function `inputs → output`. A retrieval similarity is a function of **two learned embeddings**, not raw inputs; there is no "prediction" and no label to attribute to. |
| Attention rollout / attention viz | which **tokens** a transformer attended to | Attention weights are not faithful attributions of a *downstream similarity score*; and they require the source model, not just its output vectors. |
| SAE / mech-interp toolkits (SAELens, InterPLM, TransformerLens) | **internal activations** of a specific model, for research | Built to interpret model internals offline, not to attribute a *retrieval score* between two arbitrary stored vectors at serving time, DB-agnostically. |
| Vector DB "score" + metadata filters | *that* two items matched, and any stored tags | Returns a scalar and payload; no decomposition of the score into reasons. |
| Recommender "because you watched X" | a hand-authored heuristic reason | Not derived from the actual embedding geometry; not general; not faithful. |

The **intersection** — *faithful, additive, concept-level attribution of a
similarity/ranking score, computed from vectors alone at query time, as one shipped
tool* — is **not yet occupied by a production library** (see [`EVALUATION.md`](./EVALUATION.md)
for the papers that occupy pieces of it).

### 1.2b What already exists (and why SimLens is still needed)

| Prior work | What it does | What it is **not** |
|-----------|--------------|--------------------|
| *Interpretable Text Embeddings & Text Similarity Explanation* survey (EMNLP 2025) | Maps the whole design space of similarity explanation | A tool; text-only framing |
| **S3BERT** | Decomposes similarity into fixed **semantic aspects** | Not concept-discovering; needs aspect supervision; text-only |
| **Integrated Jacobians / LRP** for similarity | Token–token interaction attribution | Needs the source model + backward passes; not serving-cheap; not concept-level |
| **COCOA** (contrastive corpus attribution) | Attributes representation similarity to **input features** via gradients + a reference corpus | Needs the model, its gradients, and raw inputs; not vector-only |
| **SAE-for-retrieval** papers (*Decoding/Disentangling Dense Embeddings*, InterPLM) | SAE features explain relevance/biology | Research methods, one level, offline, per-paper code |
| **SAE data-analysis toolkit** (arXiv 2512.10092) | SAE features for corpus analysis | Explicitly *not* ranking/similarity attribution, serving, or DB integration |
| **SAELens** | Trains/loads SAEs on model activations | Mech-interp research runtime, not a similarity/rank explainer |
| **RAGViz / RAG-E / RAG-Ex** | Explain the **generator** (attention, token saliency) | Operate above retrieval, not on embedding geometry |

SimLens's job is to be the missing **layer**: vector-only (no source model needed at
serving), multi-level under one additive contract, rank-aware, modality-agnostic, and
shipped as a fast library + sidecar with an auditable artifact — and to **interoperate**
with the above (import SAELens SAEs; borrow S3BERT aspects; fall back to Integrated
Jacobians when a model *is* available).

### 1.3 Who feels this pain

- **RAG / search engineers** debugging "why did the retriever surface this irrelevant
  chunk?" and "why did the right passage rank #7?"
- **Recommender teams** needing faithful, non-hand-wavy "why recommended" and the
  ability to steer ("less like this").
- **Regulated domains** (finance, healthcare, hiring, legal) where a retrieval or
  ranking decision must carry an auditable rationale (e.g. EU AI Act transparency
  duties).
- **ML researchers** studying what an embedding space actually encodes, and whether two
  models "agree for the same reasons."

### 1.4 Design tenets

1. **Faithful before pretty.** Every explanation must satisfy a stated mathematical
   axiom (completeness / efficiency). No post-hoc storytelling.
2. **Embedder-agnostic.** Works on any float vector from any model or modality.
3. **Store-agnostic.** Needs only vectors; integrates with any vector DB or none.
4. **Serving-time.** The hot path is a few matrix ops on static weights — fast enough to
   run per query, per hit.
5. **Auditable.** Every explanation is stamped with a content hash of the concept
   artifacts that produced it, so it is reproducible and citable.

---

## 2. Core concepts

SimLens explains a score at three **zoom levels**, all sharing one additive contract.

```
        query vector  q            candidate vector  c
                 \                        /
                  \                      /
                   ▼                    ▼
        ┌───────────────────────────────────────┐
        │              sim(q, c) = s             │   ← the opaque scalar
        └───────────────────────────────────────┘
                          │  resolve
        ┌─────────────────┴─────────────────────┐
   L1   │  dimension level   Σ φ_i  = s          │   exact, model-free
   L2   │  feature   level   Σ φ_f  = s (approx) │   SAE dictionary, monosemantic
   L3   │  concept   level   Σ φ_k  ≈ s          │   named concepts (CAV / probe)
        └───────────────────────────────────────┘
```

### 2.1 Level 1 — Dimension attribution (exact, no training)

For dot product and cosine, the score is *already* an additive function of the
coordinates. The contribution of dimension `i` is the term itself:

```
dot:     s = Σ_i q_i · c_i                      φ_i = q_i · c_i
cosine:  s = Σ_i (q_i/‖q‖)(c_i/‖c‖)             φ_i = (q_i/‖q‖)(c_i/‖c‖)
```

This is **exact** (`Σ φ_i = s`) and requires nothing but the two vectors. For Euclidean
/ squared distance there is an analogous per-dimension decomposition
(`Σ_i (q_i − c_i)² = d²`). Level 1 is the always-available baseline and the ground
truth against which the higher levels are checked.

### 2.2 Level 2 — Feature attribution (learned dictionary, monosemantic)

Raw dimensions are **polysemantic** — a single coordinate mixes several unrelated
concepts (superposition). A **sparse autoencoder (SAE)** trained on the embedding space
learns an over-complete dictionary of directions `{d_f}` such that

```
x ≈ b + Σ_f a_f(x) · d_f ,     a_f(x) = ReLU(W_enc x + b_enc)_f ,   a sparse & mostly 0
```

Each active feature `f` tends to be **monosemantic** ("this text is about litigation",
"this image contains a face", "this molecule has a sulfonamide"). Working in feature
space, the dot product decomposes over *shared active features*:

```
q·c ≈ Σ_f Σ_g a_f(q) a_g(c) (d_f · d_g)
     ≈ Σ_f a_f(q) a_f(c) ‖d_f‖²        (near-orthogonal dictionary → diagonal dominates)
```

so `φ_f = a_f(q) · a_f(c) · ‖d_f‖²` ranks the features both vectors activate. These are
the human-auditable atoms of explanation.

### 2.3 Level 3 — Concept attribution (named, supervised)

When you can supply **labeled example sets** for a concept ("relevant", "toxic", "sports
content", "kinase inhibitor"), SimLens fits a **Concept Activation Vector** (CAV): the
normal to the linear boundary separating positive from negative examples in embedding
space. A concept's contribution to a score is the projection of the interaction onto its
direction. CAVs can be:

- **derived** from your labels (TCAV-style), or
- **attached** as human-readable names to already-discovered SAE features (auto-labeling,
  §5.3), giving the best of both — unsupervised discovery + supervised naming.

### 2.4 Contrastive / rank attribution (the differentiator)

Beyond "why is A similar to q", SimLens answers **"why did A outrank B"** by attributing
the *margin*:

```
Δ = sim(q, A) − sim(q, B) = Σ_level φ^A_level − φ^B_level
```

decomposed at any zoom level. This is exactly the question search and recsys teams
actually ask, and it falls straight out of the additive contract.

### 2.5 Counterfactual steering

Because features and concepts are **directions**, SimLens supports:

- **Ablation:** "which single feature, removed from the interaction, drops this hit below
  threshold?" (minimal sufficient / necessary explanation).
- **Steering:** edit the query in concept space — `q' = q + α·d_k` — to get "more like
  this but *less* of concept k", then re-query. General across domains (recsys: "less
  sports"; search: "more recent"; vision: "less blur").

### 2.6 Non-linear / learned metrics

For learned similarity heads (a small MLP, a cross-encoder scorer, a Mahalanobis/learned
metric), the additive term trick no longer applies. SimLens falls back to **Integrated
Gradients** along the straight path from a baseline `q₀` to `q`:

```
φ_i = (q_i − q₀_i) · ∫₀¹ ∂ sim(q₀ + t(q−q₀), c) / ∂ q_i  dt
```

which satisfies **completeness** exactly: `Σ_i φ_i = sim(q, c) − sim(q₀, c)`. Same
guarantee, wider applicability.

---

## 3. Theoretical grounding

SimLens is deliberately built on results with **provable attribution axioms**, not
heuristics.

- **Additivity / efficiency (Level 1).** For dot/cosine the score is a *linear* set
  function of the coordinates. The Shapley value of a linear (inessential) game assigns
  each player exactly its own term, so the naive decomposition `φ_i = q_i c_i` *is* the
  unique fair attribution. Contributions sum to the score by construction.
- **Aumann–Shapley / Integrated Gradients (non-linear).** For differentiable non-linear
  metrics, the Aumann–Shapley cost-sharing value equals the Integrated Gradients path
  integral, which satisfies **completeness**, **sensitivity**, and **implementation
  invariance** (Sundararajan, Taly & Yan, ICML 2017).
- **Superposition & sparse dictionaries (Level 2).** Neural representations store more
  features than dimensions in linear superposition; sparse dictionary learning recovers
  approximately monosemantic directions (Elhage et al., *Toy Models of Superposition*,
  2022; Bricken et al., *Towards Monosemanticity*, 2023; Cunningham et al., *Sparse
  Autoencoders Find Highly Interpretable Features*, 2023; Templeton et al., *Scaling
  Monosemanticity*, 2024).
- **Concept directions (Level 3).** Human concepts correspond to linear directions
  learnable from example sets, and their influence on a score is measurable by
  directional derivative (Kim et al., *TCAV*, ICML 2018).
- **Faithfulness evaluation.** Explanations are validated by **deletion/insertion**
  curves and completeness residuals (Petsiuk et al., *RISE*, 2018; Hooker et al.,
  *ROAR*, 2019) — see §7.

Every shipped explanation reports its **completeness residual** `|s − Σφ|` so a consumer
can see exactly how exact it is.

---

## 4. Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  OFFLINE  (Python — simlens.train)                                     │
│  corpus of vectors ─▶ fit SAE ─▶ auto-label features ─▶ fit CAVs ─▶    │
│                              build & sign a Concept Bundle (.simlens)  │
└───────────────────────────────┬──────────────────────────────────────┘
                                 │  portable, hashed artifact
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ONLINE                                                                │
│                                                                        │
│   simlens-core (Rust)      ◀── loads Concept Bundle                    │
│     • L1 dim decomposition                                             │
│     • L2 SAE encode + feature attribution                              │
│     • L3 CAV projection                                                │
│     • IG for pluggable metrics                                         │
│     • contrastive / ablation / steering                                │
│        ▲                    ▲                     ▲                     │
│        │ PyO3               │ FFI/C-ABI           │ gRPC/HTTP           │
│   ┌────┴─────┐        ┌─────┴──────┐        ┌─────┴───────┐            │
│   │ simlens  │        │  other     │        │ simlens-serve│           │
│   │ (PyPI)   │        │  languages │        │  (sidecar)   │           │
│   └──────────┘        └────────────┘        └──────────────┘           │
└──────────────────────────────────────────────────────────────────────┘
        │ optional adapters pull vectors from:
        └─▶ Qdrant · pgvector · FAISS · Weaviate · Milvus · in-memory
```

### 4.1 Components

| Component | Language | Role |
|-----------|----------|------|
| `simlens-core` | Rust | The attribution kernels. Static weights, SIMD, allocation-light. The single source of truth for the math; every binding calls into it. |
| `simlens` | Python (PyO3/maturin wheels) | Ergonomic Python API + the **training** module (`simlens.train`) for building bundles. Published to PyPI with prebuilt manylinux/macOS/Windows wheels. |
| `simlens-serve` | Rust (`axum` + `tonic`) | Thin HTTP+gRPC server wrapping `simlens-core`, so any service/language explains vectors over the network. Stateless; loads a bundle at boot. |
| adapters | Rust + Python | Optional connectors that fetch candidate vectors by id from common stores, so SimLens drops into an existing stack with no re-plumbing. |

### 4.2 Why Rust for the core

The per-query path is: encode two vectors through an SAE (one matmul + ReLU), a handful
of dot products, and a sort. Weights are static and shared read-only. This is a classic
"small, hot, numeric" kernel — Rust gives predictable latency, SIMD, zero-copy over FFI,
and a single implementation reused by every language binding.

### 4.3 Why a Python package on top

The **offline** work (fitting SAEs and CAVs, correlating features with labels) lives
naturally in the Python/NumPy/PyTorch ecosystem, and most consumers integrate in Python.
`pip install simlens` gets both the trainer and the fast runtime via one wheel.

---

## 5. The Concept Bundle (`.simlens`)

The unit of portability and audit. A bundle is a versioned, content-hashed directory (or
single archive) describing everything needed to explain scores in one embedding space.

### 5.1 Layout

```
mybundle.simlens/
├── manifest.json          # schema version, embedder id, dim, metric, hashes, provenance
├── sae/
│   ├── encoder.safetensors     # W_enc, b_enc
│   ├── decoder.safetensors     # W_dec (dictionary d_f), b_dec
│   └── stats.json              # per-feature activation freq, sparsity, quality
├── features.json          # feature_id → { name, confidence, evidence, examples }
├── concepts/
│   ├── cavs.safetensors        # named concept directions
│   └── concepts.json           # concept_id → { name, accuracy, n_pos, n_neg }
└── labels.json            # optional raw dim → property map (Level-1 naming fallback)
```

### 5.2 `manifest.json` (illustrative)

```json
{
  "simlens_bundle_version": "1.0",
  "embedder": { "id": "some-encoder-v2", "dim": 768, "modality": "text" },
  "metric": "cosine",
  "created": "2026-07-05T00:00:00Z",
  "content_hash": "sha256:…",
  "sae": { "n_features": 16384, "l0_mean": 41.2, "expansion": 21.3 },
  "concepts": 24,
  "toolchain": { "simlens": "0.1.0" }
}
```

Every `explain()` result carries this `content_hash`, so an explanation can be tied back
to the exact artifacts that produced it — the basis of reproducibility and audit.

### 5.3 Auto-labeling features

Discovered SAE features start anonymous (`feature #2317`). SimLens names them by
correlating each feature's activation across the corpus with **user-supplied label
functions** — anything returning a scalar or boolean per item:

- text: topic/keyword/regex hits, metadata fields, classifier outputs;
- images: object-detector tags, EXIF, aesthetic scores;
- structured/numeric: any descriptor function you provide;
- generic: cluster membership, external taxonomy tags.

A feature earns a name (and a **confidence**) when its activation aligns strongly and
specifically with a label. Unnamed-but-active features are still shown as
`feature #N (unnamed)` with their nearest exemplars, never hidden — honesty over polish.

---

## 6. Public API (draft)

### 6.1 Python — offline (build a bundle)

```python
import simlens as rf

# 1. fit a monosemantic dictionary over your embedding space
sae = rf.train.SAE(dim=768, expansion=16, l1=1e-3)
sae.fit(vectors)                      # np.ndarray [N, 768] from any model

# 2. name features against your own label functions
bundle = rf.train.build_bundle(
    embedder="some-encoder-v2", metric="cosine", sae=sae,
    labelers={
        "about_finance": lambda item: keyword_score(item, FINANCE_TERMS),
        "is_recent":     lambda item: item.year >= 2024,
    },
    corpus=items, vectors=vectors,
)

# 3. optionally add supervised concepts from example sets
bundle.add_concept("relevant", positive=good_vecs, negative=bad_vecs)

bundle.save("mybundle.simlens")       # signed, hashed, portable
```

### 6.2 Python — online (explain)

```python
ex = rf.Explainer("mybundle.simlens")

# why is this candidate similar to the query?
att = ex.explain(query_vec, candidate_vec, level="feature", top_k=8)
for c in att.contributions:
    print(c.name, round(c.value, 4))          # e.g. "about_finance  0.31"
assert abs(att.completeness_residual) < 1e-5  # Σφ == score (L1) / ε (L2/L3)

# why did A outrank B?
rank_att = ex.explain_margin(query_vec, better=vec_A, worse=vec_B, level="concept")

# minimal necessary explanation
necessary = ex.ablate(query_vec, candidate_vec, threshold=0.6)

# steer the query
q2 = ex.steer(query_vec, {"about_finance": -1.0, "is_recent": +0.5})
```

### 6.3 Serving (language-agnostic)

```
POST /v1/explain
{ "query": [...], "candidates": [[...], ...], "level": "concept", "top_k": 8 }

→ { "bundle_hash": "sha256:…",
    "results": [ { "score": 0.83, "completeness_residual": 2e-6,
                   "contributions": [ {"id":"about_finance","name":"…","value":0.31}, … ] } ] }
```

Adapters let you pass candidate **ids** instead of raw vectors; SimLens fetches them from
the configured store.

### 6.4 Result object (contract)

```
Attribution
├── score:                  f64          # the original similarity
├── level:                  "dim"|"feature"|"concept"
├── contributions: [ { id, name, value, confidence, evidence? } ]  # sorted by |value|
├── completeness_residual:  f64          # |score − Σ value|  (faithfulness meter)
└── bundle_hash:            str          # provenance stamp
```

---

## 7. Faithfulness & evaluation

Explanations are only worth shipping if they are faithful. SimLens bakes evaluation in:

- **Completeness residual** on every result: `|score − Σ contributions|` (exact 0 at
  Level 1; small ε at Levels 2–3, reported honestly).
- **Deletion / insertion curves:** progressively zero out top-attributed
  features/dimensions and confirm the score drops (deletion) faster than for random
  ablations; insertion is the mirror test.
- **Necessity / sufficiency:** minimal feature set whose removal flips a threshold
  decision, and minimal set that alone reproduces the match.
- **SAE health metrics:** L0 sparsity, dead-feature rate, reconstruction loss, and
  feature-naming coverage/confidence — surfaced in `stats.json`.
- **Cross-run stability:** re-training a bundle should yield semantically stable named
  concepts (report drift).

A `simlens eval` CLI runs these against a held-out set and emits a scorecard.

---

## 8. Example use cases (domain-neutral)

- **RAG debugging.** "Why did the retriever surface this chunk, and why did the correct
  passage rank below it?" → margin attribution over named topical features.
- **Recommender transparency & control.** Faithful "because it shares X, Y with things
  you liked" + a user-facing "less like this" steer.
- **Search relevance tuning.** Discover that a query is matching on a spurious feature
  (boilerplate, formatting) and ablate it.
- **Model comparison / embedding audits.** Do two encoders rank the same pair similar
  *for the same reasons*? Compare their concept spectra.
- **Regulated decisioning.** Attach an auditable, hash-stamped rationale to any
  retrieval- or similarity-driven decision.
- **Dataset / drift monitoring.** Track which concepts dominate retrieval over time.

---

## 9. Roadmap

| Phase | Deliverable |
|-------|-------------|
| **P0 — Kernel** | `simlens-core` (Rust): Level-1 exact dim attribution for dot/cosine/euclidean; contrastive margin; completeness checks. PyO3 wheel. `explain()` end to end on in-memory vectors. |
| **P1 — Dictionaries** | `simlens.train.SAE` + bundle format + auto-labeling; Level-2 feature attribution in the core; `simlens eval` faithfulness scorecard. |
| **P2 — Concepts** | CAV fitting, Level-3 concept attribution, steering & ablation APIs. |
| **P3 — Non-linear** | Integrated-Gradients path for pluggable/learned metrics via a scorer callback. |
| **P4 — Serving & adapters** | `simlens-serve` (HTTP+gRPC) + Qdrant/pgvector/FAISS adapters; language-agnostic clients. |
| **P5 — Hardening** | Prebuilt wheels (manylinux/macOS/Windows) + crate publish; benchmarks; docs site; example notebooks per domain. |
| **P6 — Multi-vector** | Late-interaction / token-pair attribution (§13.10) + alignment-heatmap renderer. |

> Ease-of-use gates: **P0 must ship §13.1 zero-setup Level-1 mode** and **P1 must ship
> §13.9 `import_sae`** so the tool delivers value before anyone trains anything. These are
> the highest-leverage adoption levers per [`EVALUATION.md`](./EVALUATION.md).

## 10. Non-goals (for now)

- Not a vector database, embedder, or reranker — it explains their output, it does not
  replace them.
- Not a global model-interpretability suite for arbitrary neural nets — it is scoped to
  **similarity/ranking over embeddings**.
- Not a UI product — it is a library + sidecar; visualization is left to consumers
  (though example renderers may ship).

## 11. Open questions

- **Metric coverage** for Level-2/3 beyond dot/cosine (learned metrics rely on the IG
  path — acceptable latency budget?).
- **Bundle signing**: content hash only, or optional cryptographic signature for
  regulated deployments?
- **SAE trainer scope**: ship a minimal built-in trainer, or also adapt externally
  trained SAEs (import path)?
- **Multi-vector / late-interaction** models (e.g. token-level retrieval): a natural but
  larger extension — decompose over token pairs.

---

## 12. Package schema design

This section pins down the concrete shape of the codebase, the module boundaries, and the
typed data contracts, so implementation can start without further design churn.

### 12.1 Repository layout (Cargo workspace + maturin)

A single mixed Rust/Python monorepo. Rust owns the math and the hot path; Python owns
training, evaluation, adapters, and ergonomics. One wheel ships both.

```
simlens/
├── Cargo.toml                     # workspace manifest
├── pyproject.toml                 # maturin backend → builds the wheel
├── crates/
│   ├── simlens-core/              # pure attribution kernels, NO I/O, no_std-friendly
│   │   └── src/{lib,metric,level1,level2,level3,contrastive,ig,
│   │            ablate,steer,sae,cav,types,linalg}.rs
│   ├── simlens-bundle/            # bundle load/save/verify (safetensors + hashing)
│   │   └── src/{lib,manifest,sae_io,cav_io,hash,registry}.rs
│   ├── simlens-serve/             # axum (HTTP) + tonic (gRPC) server binary
│   │   └── src/{main,http,grpc,state,config}.rs
│   └── simlens-py/                # PyO3 glue crate → compiled into python/simlens/_native
│       └── src/lib.rs
├── python/simlens/
│   ├── __init__.py                # re-exports: Explainer, Bundle, train, eval, adapters
│   ├── _native/                   # compiled extension (from simlens-py)
│   ├── explain.py                 # Explainer facade over the native core
│   ├── types.py                   # typed dataclasses mirroring core types
│   ├── train/{sae,cav,labeling,bundle,__init__}.py
│   ├── eval/{faithfulness,scorecard,cli,__init__}.py
│   ├── adapters/{base,qdrant,pgvector,faiss,weaviate,memory,__init__}.py
│   └── viz/{highlight,render_text,__init__}.py
├── clients/                       # thin gRPC clients for other languages (ts, go)
├── proto/simlens.proto            # the serving contract, single source of truth
├── examples/                      # runnable notebooks per domain (text, image, recsys)
├── benches/                       # criterion + pytest-benchmark
└── docs/
```

**Dependency direction (must stay acyclic):**
`simlens-core` ← `simlens-bundle` ← {`simlens-py`, `simlens-serve`}. `core` never depends
on `bundle` (it takes already-loaded weights as plain slices), which keeps the math unit
independently testable and `no_std`-friendly.

### 12.2 Core type schema (language-neutral)

These types are defined once in `simlens-core/src/types.rs`, mirrored 1:1 in
`python/simlens/types.py` (dataclasses) and `proto/simlens.proto` (messages). A
contract test asserts the three stay in sync.

```rust
enum Metric { Dot, Cosine, Euclidean, Custom(ScorerId) }
enum Level  { Dim, Feature, Concept }

struct Contribution {
    id:         ContribId,        // dim index | feature id | concept id
    name:       Option<String>,   // human name if known (None = unnamed feature)
    value:      f64,              // signed contribution to the score
    confidence: Option<f64>,      // naming/probe confidence in [0,1]
    polarity:   Polarity,         // Shared | QueryOnly | CandidateOnly  (see §13.4)
    evidence:   Option<Evidence>, // exemplars / nearest items for this concept
}

struct Attribution {
    score:                 f64,
    metric:                Metric,
    level:                 Level,
    contributions:         Vec<Contribution>,   // sorted by |value| desc
    completeness_residual: f64,                 // |score − Σ value|
    coverage:              f64,                  // fraction of score mass named  [0,1]
    bundle_hash:           Option<String>,       // provenance stamp (None at Level 1)
    warnings:              Vec<Warning>,          // honesty flags (§13.7)
}

struct ExplainConfig {
    level:            Level,
    top_k:            usize,
    min_abs:          f64,     // drop contributions below this |value|
    include_polarity: bool,    // compute dissimilarity contributions too
    normalize:        Normalize, // Raw | ScoreFraction (values as % of score)
}
```

Python mirror (what users touch):

```python
@dataclass(frozen=True)
class Contribution:
    id: str; name: str | None; value: float
    confidence: float | None; polarity: Polarity; evidence: Evidence | None

@dataclass(frozen=True)
class Attribution:
    score: float; metric: str; level: str
    contributions: list[Contribution]
    completeness_residual: float; coverage: float
    bundle_hash: str | None; warnings: list[str]
    def as_sentence(self) -> str: ...          # deterministic NL rendering (§13.6)
    def to_dict(self) -> dict: ...             # stable JSON for logging/audit
```

### 12.3 Bundle on-disk schema (JSON Schema, abbreviated)

The `.simlens` bundle from §5 is validated against a published JSON Schema so third
parties can produce compatible bundles.

```jsonc
// manifest.schema.json  (draft 2020-12)
{
  "type": "object",
  "required": ["simlens_bundle_version","embedder","metric","content_hash"],
  "properties": {
    "simlens_bundle_version": { "const": "1.0" },
    "embedder": { "type":"object",
      "required":["id","dim","modality"],
      "properties":{ "id":{"type":"string"}, "dim":{"type":"integer"},
                     "modality":{"enum":["text","image","audio","structured","other"]} } },
    "metric": { "enum": ["dot","cosine","euclidean"] },
    "sae":    { "type":"object", "required":["n_features"],
                "properties":{ "n_features":{"type":"integer"},
                               "l0_mean":{"type":"number"},
                               "expansion":{"type":"number"} } },
    "concepts": { "type":"integer" },
    "content_hash": { "type":"string", "pattern":"^sha256:[0-9a-f]{64}$" },
    "signature": { "type":["string","null"] },        // optional (§13.8)
    "created": { "type":"string","format":"date-time" },
    "toolchain": { "type":"object" }
  }
}
```

Weights (`encoder`, `decoder`, `cavs`) are `safetensors` — language-portable, zero-copy,
mmap-able by the Rust core. `features.json` / `concepts.json` carry names, confidences,
and exemplar ids.

### 12.4 Serving contract (`proto/simlens.proto`, abbreviated)

```proto
service SimLens {
  rpc Explain       (ExplainRequest)  returns (ExplainResponse);
  rpc ExplainMargin (MarginRequest)   returns (ExplainResponse);
  rpc Steer         (SteerRequest)     returns (Vector);
  rpc BundleInfo    (BundleRef)        returns (Manifest);
}
message ExplainRequest {
  repeated float query = 1;
  repeated Vector candidates = 2;      // OR candidate_ids + adapter fetch
  repeated string candidate_ids = 3;
  string level = 4; uint32 top_k = 5; string bundle = 6;
}
```

HTTP is a 1:1 JSON transcoding of the same messages (§6.3).

### 12.5 Public Python surface (what `import simlens` exposes)

```
simlens
├── Explainer(bundle=None, metric="cosine")         # None ⇒ Level-1-only, zero-setup
│    ├── .explain(q, c, level, top_k, ...) -> Attribution
│    ├── .explain_batch(q, cs) -> list[Attribution]
│    ├── .explain_margin(q, better, worse, level) -> Attribution
│    ├── .explain_vs_corpus(q, c, corpus, foil) -> Attribution   # COCOA-style (§13.2)
│    ├── .ablate(q, c, threshold) -> AblationResult
│    ├── .steer(q, {concept: weight}) -> np.ndarray
│    └── .summarize(q, hits) -> CohortSummary                    # shared concepts (§13.3)
├── Bundle.load(path) / .save(path) / .verify()
├── train.{SAE, fit_cav, build_bundle, import_sae}               # import_sae ⇐ SAELens/HF
├── eval.{faithfulness, scorecard}
├── adapters.{Qdrant, Pgvector, Faiss, Weaviate, Memory}
└── viz.{highlight, render}
```

---

## 13. Extended & improved functionalities

Beyond the P0–P5 core, these are the capabilities that raise SimLens from "a nice SAE
wrapper" to a differentiated tool. Each is tied to a prior-art gap identified in
[`EVALUATION.md`](./EVALUATION.md).

### 13.1 Zero-setup Level-1 mode *(ease-of-use, critical)*
`Explainer()` with **no bundle** immediately returns exact dimension attributions and
margin decompositions. This kills the "must train an SAE before any value" barrier —
users get a faithful, if raw, explanation on line one, then opt into concepts later.

### 13.2 Contrastive-corpus explanation (COCOA-inspired)
`explain_vs_corpus(q, c, corpus, foil)` grounds the explanation in *what makes `c` similar
to `q` relative to a reference corpus and a foil set*, which yields far more meaningful
concept selection than explaining a bare pair — directly borrowing the COCOA insight but
working **vector-only** (no source-model gradients required).

### 13.3 Cohort / result-set summarization
`summarize(q, hits)` aggregates attributions across a whole result page: *"8 of these 10
results matched primarily on concept X; results 4 and 7 additionally share Y."* This is
the view search/recsys operators actually want, and no existing tool offers it.

### 13.4 Dissimilarity & "why not" attribution
Every contribution carries a **polarity** (`Shared` / `QueryOnly` / `CandidateOnly`).
This lets SimLens answer the survey's open "why was a document *omitted*?" question:
surface concepts strong in the query but absent in the candidate (and vice-versa).

### 13.5 Aspect grouping (S3BERT-style) & concept hierarchy
Features → named concepts → coarse **aspects** (a small user-defined taxonomy, e.g.
"topic / tone / recency"). `explain(..., level="aspect")` reports the score split across
aspect buckets — a compact, executive-readable view that composes with the detailed ones.

### 13.6 Deterministic natural-language rendering
`Attribution.as_sentence()` composes contributions into a readable sentence from a
template ("*Matched mainly on ‘financial regulation’ (61%) and ‘2024 filings’ (22%)*").
Deterministic, no LLM, fully faithful to the numbers — improves UX without sacrificing
auditability. (An optional LLM-polish hook is available but clearly tagged non-faithful.)

### 13.7 Always-on honesty flags
`Attribution.warnings` raises when: completeness residual exceeds a threshold (auto
**downgrade to Level 1** and say so); coverage is low (much score mass is unnamed);
metric is unsupported for the requested level; or the bundle's embedder id / dim doesn't
match the vectors. Never silently pretty-print an untrustworthy explanation.

### 13.8 Audit mode
`content_hash` on every result, optional bundle **signature**, and a hash-addressable
explanation cache so identical (query, candidate, bundle) always yields byte-identical
output — the substrate for reproducible, regulator-facing decision logs.

### 13.9 Interop & import (don't reinvent)
`train.import_sae()` adapts an externally trained SAE (SAELens / HF safetensors) into a
bundle; adapters read vectors straight from popular stores; the serving proto gives Go/TS
clients first-class access. SimLens is a **layer**, not a walled garden.

### 13.10 Late-interaction / multi-vector attribution *(roadmap P6)*
For token-level retrievers (ColBERT-style), attribute the score over **token-pair
alignments** (Integrated-Jacobian / BERTScore lineage), rendered as an alignment heatmap.
The natural extension once single-vector attribution is solid.

### 13.11 Calibrated confidence
Feature/concept naming confidence is **calibrated** on a held-out split (reliability
curve in the scorecard), so a `confidence=0.8` means what it says and thresholds behave
predictably across bundles.

---

## Appendix A — References

- Sundararajan, Taly, Yan. *Axiomatic Attribution for Deep Networks (Integrated
  Gradients).* ICML 2017.
- Kim et al. *Interpretability Beyond Feature Attribution: TCAV.* ICML 2018.
- Elhage et al. *Toy Models of Superposition.* Anthropic, 2022.
- Bricken et al. *Towards Monosemanticity: Decomposing Language Models with Dictionary
  Learning.* Anthropic, 2023.
- Cunningham et al. *Sparse Autoencoders Find Highly Interpretable Features in Language
  Models.* 2023.
- Templeton et al. *Scaling Monosemanticity.* Anthropic, 2024.
- Petsiuk et al. *RISE: Randomized Input Sampling for Explanation.* BMVC 2018.
- Hooker et al. *A Benchmark for Interpretability Methods (ROAR).* NeurIPS 2019.
- Shapley. *A Value for n-Person Games.* 1953. / Aumann & Shapley. *Values of Non-Atomic
  Games.* 1974.

**Similarity-explanation prior art (see `EVALUATION.md` for how SimLens relates):**

- *Interpretable Text Embeddings and Text Similarity Explanation: A Survey.* arXiv
  2502.14862 (EMNLP 2025) — the field map.
- Opitz & Frank. *SBERT studies Meaning Representations (S3BERT).* — aspect-wise
  similarity decomposition.
- *Contrastive Corpus Attribution (COCOA) for Explaining Representations.* arXiv
  2210.00107.
- *Decoding Dense Embeddings: Sparse Autoencoders for Interpreting and Discretizing Dense
  Retrieval.* arXiv 2506.00041; *Disentangling Dense Embeddings with Sparse
  Autoencoders.* arXiv 2408.00657.
- *Interpretable Embeddings with Sparse Autoencoders: A Data Analysis Toolkit.* arXiv
  2512.10092.
- InterPLM. *Discovering Interpretable Features in Protein Language Models via SAEs.*
  bioRxiv 2024; PNAS 2025.
- SAELens — sparse-autoencoder training/inference library.
- Integrated Jacobians / LRP for text-similarity interaction attribution (surveyed in
  2502.14862).
