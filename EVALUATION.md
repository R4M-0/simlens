# SimLens — Critical Evaluation

> A red-team of the [SPEC](./SPEC.md): does SimLens answer a *real* problem, is it
> *distinct* from existing work, and would it be *usable*? This document tests the design
> against (A) the literature and existing tools, and (B) UX, practicality, customization,
> and ease of use. Findings feed back into the spec; the highest-severity ones are
> already reflected in SPEC §13.
>
> Method: prior-art search (arXiv/EMNLP/GitHub, July 2026), method-by-method comparison,
> and a heuristic developer-experience walkthrough of concrete scenarios. Nothing here is
> benchmarked against a running build yet — this is a *design* evaluation.

---

## Part A — Test against literature & existing tools

### A.1 The field is real and active (thesis correction)

The original pitch claimed the problem space was "empty." **That is false and has been
corrected in the spec.** Explaining embedding similarity is an active research area with a
2025 EMNLP survey devoted to it. SimLens must be justified as *engineering + unification +
guarantees + generality*, not novelty of the core idea.

### A.2 Competitive map

| System | Level | Needs source model? | Serving-time? | Rank/margin? | Modality | Shipped as | Overlap with SimLens |
|--------|-------|--------------------|---------------|--------------|----------|-----------|----------------------|
| **S3BERT** (aspect decomposition) | concept (fixed aspects) | yes (fine-tuned) | ~ | no | text | model | Concept/aspect view (§13.5) — SimLens discovers rather than fixes aspects |
| **Integrated Jacobians / LRP** | token interactions | **yes (gradients)** | costly (2·d passes) | no | text | method | SimLens's non-linear fallback (§2.6); SimLens avoids it when vectors suffice |
| **COCOA** | input features | **yes (gradients)** | costly | no | text/image | method | Contrastive grounding (§13.2) — SimLens does it **vector-only** |
| **SAE-for-retrieval** (Decoding/Disentangling Dense Embeddings, InterPLM) | features | no (post-hoc on vectors) | offline analysis | no | text/protein | per-paper code | **Direct overlap** with Level 2 — this is the strongest prior art |
| **SAE data-analysis toolkit** (2512.10092) | features | no | offline | no | text | library (research) | Closest *tool*; explicitly not ranking/serving/DB |
| **SAELens** | model features | yes (activations) | offline/research | no | model internals | library | SimLens *imports* its SAEs (§13.9); different goal |
| **RAGViz / RAG-E / RAG-Ex / XGRAG** | generator attention/tokens | yes (LLM) | debugging | n/a | text | tools | Different layer (generation, not retrieval geometry) |
| **BERTScore / ColBERT alignment** | token alignment | yes (model) | ~ | no | text | metric/model | SimLens's P6 multi-vector path (§13.10) |
| **SHAP / LIME / Captum** | prediction from inputs | yes | costly | no | any | libraries | Not similarity/rank at all — adjacent, not competing |

### A.3 Where SimLens is genuinely differentiated

1. **Vector-only at serving.** COCOA, Integrated Jacobians, S3BERT all need the source
   model (and often gradients and raw inputs). SimLens explains from the **stored vectors
   alone** — the only thing a production vector DB actually has. This is the load-bearing
   distinction and it holds up.
2. **Rank/margin attribution as a first-class operation.** No surveyed system decomposes
   *"why A outranked B."* Yet that is the question retrieval/recsys teams ask daily. This
   is SimLens's clearest white space.
3. **One additive contract across three zoom levels** with a **reported completeness
   residual** on every call. The literature offers pieces (aspects *or* features *or*
   token interactions); none unifies them under a stated faithfulness invariant that
   ships in the output.
4. **Modality-agnostic + production packaging** (Rust hot path, DB adapters, gRPC sidecar,
   signed audit bundle). Existing work is overwhelmingly text-only research code.
5. **Dissimilarity / "why not" attribution** (§13.4) addresses an *explicit open problem*
   the EMNLP survey names ("why was a document omitted?"). No tool does this.

### A.4 Where the thesis is weakest (honest risks)

- **R1 — Level 2 is not novel.** SAE-on-retrieval-embeddings is published multiple times.
  SimLens's Level 2 is *productization*, not invention. Mitigation: lead with rank
  attribution + vector-only + unification + serving, treat SAEs as table stakes, and
  **interoperate** (import SAELens/HF SAEs) rather than compete on SAE quality.
- **R2 — The "near-orthogonal dictionary → diagonal dominates" step (SPEC §2.2) is an
  approximation, not a theorem.** SAE decoder atoms are *not* guaranteed orthogonal;
  cross terms `a_f(q)·a_g(c)·(d_f·d_g)` can be non-negligible, so Level-2 completeness is
  approximate and the residual can be large for some pairs. Mitigation: this is exactly
  why the **completeness residual is mandatory in the output** and why §13.7 auto-downgrades
  to the exact Level 1 when the residual is too big. Do **not** market Level 2 as "exact."
- **R3 — IG completeness for cosine is baseline-sensitive.** Integrated Gradients sums to
  `sim(q,c) − sim(q₀,c)`, not to `sim(q,c)`; the choice of baseline `q₀` (zero vector?
  mean vector?) changes the story and a zero baseline is degenerate for cosine. Mitigation:
  document baseline semantics explicitly; default to a mean/centroid baseline and report
  it in the result. Don't claim unconditional completeness for non-linear metrics.
- **R4 — Naming is the real bottleneck, not attribution.** Ranking dimensions/features is
  easy and exact; turning feature #2317 into a trustworthy human label is the hard,
  error-prone part, and bad labels are worse than none (confident nonsense). Mitigation:
  calibrated confidence (§13.11), always show unnamed-but-active features honestly,
  exemplar evidence per concept, and coverage reporting.
- **R5 — "General-purpose" cuts both ways.** A tool that works for any embedding gives up
  the domain priors (RDKit descriptors, Pfam, ontology tags) that make labels *good*. The
  labeler API (§5.3) is how domain knowledge re-enters, but out-of-the-box quality on an
  unknown space will be modest. Set expectations accordingly.

### A.5 Verdict on Part A

The problem is **real and unsolved as a product**, but **partially solved in research**.
SimLens's defensible position is *the missing production layer*: vector-only, rank-aware,
multi-level, modality-agnostic, auditable, and interoperable. Drop every "first to…"
claim; keep every "only shipped tool that…" claim and make sure each is actually true.

---

## Part B — Test against UX, practicality, customization, ease of use

Scenario-based heuristic evaluation. Severity: 🔴 blocker · 🟠 major · 🟡 minor.

### B.1 Practicality — "can I install and run it?"

- ✅ Rust core + prebuilt wheels means `pip install simlens` with no toolchain — good.
- 🔴 **Time-to-first-value was gated on training an SAE.** A user with vectors but no
  bundle got *nothing*. This is the single biggest adoption killer for an interpretability
  tool. → **Fixed in SPEC §13.1**: `Explainer()` with no bundle returns exact Level-1
  attribution and margin decomposition instantly. First value on line one.
- 🟠 **Pretrained bundles needed.** Even with Level 1, the "wow" is concepts, which need a
  bundle. → Recommendation (now in roadmap intent): ship/host downloadable bundles for a
  handful of popular public embedders (e5, bge, all-MiniLM, CLIP) so most users skip
  training entirely. Tracked as an open item — see B.6.
- 🟡 Serving sidecar adds an ops surface. Acceptable because it's optional; the in-process
  library path exists for those who don't want another service.

### B.2 Ease of use — "is the happy path short and honest?"

- ✅ Core call is one line: `ex.explain(q, c)`; sensible defaults (`level="feature"` if a
  bundle exists else `"dim"`, `top_k=8`, `metric="cosine"`).
- ✅ `Attribution.as_sentence()` (§13.6) gives a readable answer without the caller doing
  presentation work — big for casual users and demos.
- 🟠 Output could overwhelm: 16k features, most near-zero. → Enforced by `top_k` +
  `min_abs` defaults and `coverage` so the user sees "these 8 explain 84% of the score."
- 🟠 Three levels + polarity + aspects is conceptually heavy. → Progressive disclosure:
  the default result is a short list + sentence; levels/polarity/aspects are opt-in
  parameters, not mandatory reading.
- 🟡 Risk of over-trust in confident labels (R4). → Warnings + calibrated confidence +
  visible residual make distrust the *default* posture when the numbers are shaky.

### B.3 Customization — "can I make it fit my domain/stack?"

- ✅ **Labelers** (§5.3) accept arbitrary user functions → any domain taxonomy plugs in.
- ✅ **Custom metric** via a scorer callback (IG path) → learned/exotic similarities work.
- ✅ **Bring-your-own SAE** (`import_sae`, §13.9) → interop with SAELens/HF, no lock-in.
- ✅ **Adapters** for Qdrant/pgvector/FAISS/Weaviate/in-memory → drops into existing
  stacks; `Memory` adapter means zero-infra local use.
- ✅ **Aspect taxonomy** (§13.5) user-defined → executive vs analyst views.
- 🟡 Bundle format is opinionated (safetensors + JSON). Mitigated by publishing the JSON
  Schema (SPEC §12.3) so third parties can emit compatible bundles.
- 🟠 No obvious story yet for **fine-tuned/rotating embedders** (bundle goes stale when the
  encoder is retrained). → Recommendation: bundle carries `embedder.id`; registry
  refuses/ warns on mismatch (§13.7); document a re-fit workflow + drift report (§13.11).

### B.4 Interpretability-of-the-output — "will a human trust and act on it?"

- ✅ Completeness residual + coverage make faithfulness legible, not assumed.
- ✅ Polarity ("shared / query-only / candidate-only") maps to how people reason about
  similarity and difference.
- ✅ Per-concept **evidence exemplars** let a user sanity-check a label ("show me items
  that max-activate feature #2317").
- 🟠 Numbers need anchoring. A raw contribution of `0.31` is meaningless alone. →
  `Normalize=ScoreFraction` renders contributions as % of the score by default in the
  sentence view.

### B.5 Developer experience scorecard

| Dimension | Score | Rationale |
|-----------|:----:|-----------|
| Install / setup | 4 / 5 | Wheels are easy; pretrained bundles still needed for full value (B.1). |
| Time-to-first-value | 4 / 5 | Now instant at Level 1 (§13.1); concepts still need a bundle. |
| Happy-path ergonomics | 4 / 5 | One-liner + NL sentence; some conceptual surface area. |
| Customization | 5 / 5 | Labelers, custom metrics, BYO-SAE, adapters, aspects. |
| Interop / no lock-in | 4 / 5 | Imports external SAEs; DB adapters; gRPC clients. Bundle format opinionated. |
| Output trustworthiness | 4 / 5 | Residual/coverage/warnings/evidence strong; naming quality is the residual risk (R4). |
| Docs/onboarding (planned) | — | Not built; per-domain notebooks are in the roadmap (P5). |
| **Overall** | **4.2 / 5** | Strong *if* zero-setup mode and pretrained bundles land early. |

### B.6 Prioritized recommendations (severity-ordered)

1. 🔴 **Zero-setup Level-1 mode** — *done in spec (§13.1), gated into P0.*
2. 🟠 **Ship pretrained bundles for ~5 popular public embedders** — biggest remaining
   time-to-value lever. Add as an explicit P1 deliverable. *(Open — not yet in roadmap.)*
3. 🟠 **`import_sae` from SAELens/HF** — *done (§13.9), gated into P1.*
4. 🟠 **Mandatory residual + auto-downgrade + warnings** — *done (§13.7).* Guards R2/R3.
5. 🟠 **Calibrated, evidence-backed naming** — *done (§13.11 + evidence field).* Guards R4.
6. 🟡 **Embedder-version / drift handling** — partially covered (registry + `embedder.id`);
   flesh out the re-fit workflow in docs.
7. 🟡 **Baseline semantics for IG** documented and defaulted to centroid — add to P3 notes.

---

## Part C — Overall verdict

**Problem:** real, painful, and unsolved *as a product*. ✅
**Distinctiveness:** solid once reframed — vector-only, rank-aware, unified, modality-
agnostic, auditable, interoperable — but **not** novel at the level of SAEs-on-embeddings
(R1). Positioning must be disciplined. ⚠️→✅
**Usability:** strong after the zero-setup fix; the remaining lever is pretrained bundles.
✅ (contingent on B.6 #2)

**Single biggest risk:** label quality (R4) — the tool lives or dies on whether
`feature #2317 = "aromatic sulfonamide"` is *right*. Every honesty mechanism in §13.7 /
§13.11 exists to make wrong labels visible instead of authoritative. Treat naming quality,
not attribution math, as the primary research investment.

**Recommended one-line positioning:**
> *SimLens is the faithful, vector-only explanation layer for similarity and ranking —
> the "why" that vector search never returns — usable with any embedder and any store.*
