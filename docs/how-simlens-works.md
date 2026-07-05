# How SimLens works

*A guided tour for the curious. No prior background in interpretability needed — just a
rough idea of what an "embedding" is.*

---

## The problem SimLens solves

Modern search, recommendation, and retrieval systems turn everything — a sentence, an
image, a product, a molecule — into a list of numbers called an **embedding vector**.
Two things are considered "similar" when their vectors point in a similar direction. A
database compares them and hands you back a single number: `0.83`.

That number tells you *that* two things are similar. It never tells you **why**.

For a lot of applications that's fine. But the moment a human has to *trust* or *act on*
the result — a doctor looking at "these two case reports are related," an analyst asking
"why did the search rank this document first," a shopper wondering "why was this
recommended" — the bare score is frustrating. You want the reasons, in words.

**SimLens produces those reasons.** Given the two vectors and the similarity metric, it
breaks the score down into the specific ingredients that produced it, and — crucially —
it can *prove* the breakdown adds up.

---

## The big idea: a similarity score is a sum

Here's the insight everything rests on. The most common way to compare two vectors is the
**dot product**: multiply them coordinate by coordinate, then add up the results.

```
score = q₁·c₁ + q₂·c₂ + q₃·c₃ + … + q₇₆₈·c₇₆₈
```

Look at that: the score is *literally already a sum* of per-coordinate pieces. Each term
`qᵢ·cᵢ` is the exact amount that coordinate *i* contributed to the final number. Nothing
is hidden. If coordinate 412 contributed 0.44 out of a 0.55 score, then coordinate 412 is
80% of the reason these two things matched — and that's not an estimate, it's arithmetic.

**Cosine similarity** (the other popular metric) is the same idea after scaling both
vectors to length 1, so it decomposes exactly the same way. This gives SimLens its
foundation: for the metrics almost everyone uses, an *exact, honest* explanation of a
similarity score already exists inside the math. SimLens surfaces it.

This is **Level 1**, and it needs no training, no model, no setup — just the two vectors.

```python
import simlens
ex = simlens.Explainer(metric="cosine")
attr = ex.explain(query_vec, candidate_vec)
print(attr.as_sentence())
# → "Matched mainly on 'dim:0' (80%), 'dim:3' (20%)."
```

---

## The honesty dial: the completeness residual

Any explanation tool can *claim* to tell you why. The question is whether the explanation
is **faithful** — whether it actually reflects the math, or is just a plausible story.

SimLens attaches a number to every explanation called the **completeness residual**: the
gap between the real score and the sum of the contributions it reported.

```
residual = | score − (sum of all contributions) |
```

- At Level 1, this residual is **zero** (to floating-point precision). The explanation
  *is* the arithmetic; it cannot be wrong.
- At the higher levels (below), the residual can be larger — and SimLens **reports it,
  every time**, instead of hiding it. If the residual gets big, SimLens raises a warning
  and tells you to drop back to the exact level.

This is the core design principle: **never present a pretty explanation you can't stand
behind.** The residual is always in the box.

---

## Zooming in: three levels of explanation

"Coordinate 412 contributed 80%" is faithful, but coordinate 412 doesn't *mean* anything
to a human. SimLens offers three **zoom levels**, trading a little exactness for a lot of
readability as you go up.

### Level 1 — Dimensions (exact, zero setup)

The raw per-coordinate breakdown described above. Always available, always exact. Great
for debugging and as the ground truth the other levels are checked against. Its weakness:
individual embedding coordinates are usually **polysemantic** — one coordinate mixes
several unrelated ideas together — so the names ("dim:412") aren't meaningful.

### Level 2 — Features (learned, monosemantic)

To get meaningful units, SimLens learns a **dictionary** of directions in the embedding
space using a small neural network called a **sparse autoencoder (SAE)**. The SAE is
trained to reconstruct embeddings using only a handful of "features" active at a time.
Because of that sparsity pressure, the features it discovers tend to be **monosemantic** —
each one lights up for a single, coherent concept ("this text is about litigation," "this
image contains a face," "this molecule has a sulfonamide group").

SimLens then measures which features *both* vectors activate, and how much each shared
feature drove the match:

```
contribution of feature f  =  (how strongly q activates f)
                            × (how strongly c activates f)
                            × (the feature's weight in the space)
```

Now the explanation reads in terms of features instead of anonymous coordinates. And
because features are auto-labeled (next section), many of them come with human names.

Level 2 is an *approximation* of the score — so its residual is usually non-zero, and
SimLens shows it honestly.

### Level 3 — Concepts (named, from your examples)

Sometimes you already know the concepts you care about ("relevant," "toxic," "sports
content," "kinase inhibitor"). If you can supply a handful of positive and negative
example vectors for a concept, SimLens fits a **concept direction** (a Concept Activation
Vector, a technique from the interpretability literature) and can then decompose any score
along your named concepts.

Concepts are a *chosen subspace*, not a complete basis, so Level-3 explanations are
deliberately **partial** — SimLens says so in a warning and reports how much of the score
the named concepts actually cover.

Here's the same pair of items explained at Levels 2 and 3 (from the bundled demo):

```
feature attribution  score=0.582  residual=2.2e-01  coverage=90%
  concept_4   +0.1832 ~0.98  ████████████████████████████
  concept_2   +0.1009 ~0.97  ███████████████
  ⚠ completeness_residual_high: Σφ deviates from score by 0.22 (38%); use level="dim"…

concept attribution  score=0.582  residual=5.1e-02  coverage=99%
  concept_4   +0.3067 ~1.00  ████████████████████████████
  concept_2   +0.2539 ~1.00  ███████████████████████
  concept_5   −0.0322 ~1.00  ░░░
  ⚠ partial_decomposition: concepts span a subspace, not a complete basis…
```

The `~0.98` is the label's **confidence**; the bars are signed (solid = pushes similarity
up, shaded = pushes it down); and the warnings are the honesty dial doing its job.

---

## Shared, or one-sided? (polarity)

Every contribution is tagged with a **polarity**:

- **shared** — both items have this feature (it's a *reason they match*),
- **query-only** — the query has it but the candidate doesn't,
- **candidate-only** — the reverse.

This lets SimLens answer not just "why are these similar" but the mirror question **"why
aren't they *more* similar / why was something omitted"** — you look at the strong
query-only features the candidate is missing.

---

## Beyond a single pair

The additive foundation makes several genuinely useful operations fall out for free.

### Why did A outrank B? (margin attribution)

In search and recommendation, the real question is rarely "why does A match" — it's **"why
did A beat B?"** Because scores are sums, the *difference* between two scores is also a
sum, so SimLens can decompose the **margin** directly:

```python
ex.explain_margin(query, better=A, worse=B)
# → "Matched mainly on 'concept_4' (52%), 'concept_2' (33%), 'concept_3' (−15%)."
```

Positive terms are why A won; negative terms are places B was actually stronger.

### What's the minimal reason? (ablation)

SimLens can greedily find the **smallest set of features** whose removal would drop the
match below a threshold — the "if you only remember one thing" explanation.

```python
abl = ex.ablate(query, candidate, threshold=0.5)
# → removing 1 feature: 0.582 → 0.399 (dropped_below=True)
```

### "More like this, but less of that" (steering)

Because concepts are directions, you can **edit a query** in concept space and search
again — "more like this document but less about sports," "more recent," "less lipophilic."

```python
new_query = ex.steer(query, {"sports": -1.0, "finance": +0.5})
```

### Other views

- **Contrastive-corpus** explanations describe what makes a hit similar *relative to* a
  background set, which sharpens the concepts that get surfaced.
- **Aspect view** rolls up fine-grained concepts into a few big buckets ("topic / tone /
  recency") for an at-a-glance summary.
- **Cohort summary** aggregates over a whole result page: "8 of these 10 results matched
  mainly on concept X."

---

## Bundles: portable, auditable artifacts

The learned dictionary (the SAE) and the named concepts are packaged into a **bundle** — a
small directory you train once and reuse. A bundle is stamped with a **content hash**, and
every explanation it produces carries that hash. That means an explanation is
**reproducible and traceable**: you can prove which exact artifacts produced a given
rationale — useful anywhere a decision has to be defensible.

```python
bundle.save("mybundle.simlens")
ex = simlens.Explainer("mybundle.simlens")   # load and go
```

Bundles are also how you bring your own dictionary: if you've trained a sparse autoencoder
elsewhere, SimLens can import it rather than making you retrain.

---

## Works with anything

Two deliberate choices make SimLens broadly usable:

- **Embedder-agnostic.** It only ever touches the vectors, so it works with any embedding
  model and any modality — text, images, audio, structured data, molecules.
- **Store-agnostic.** It doesn't replace your vector database; it explains its output.
  Adapters can pull candidate vectors from common stores, and a small HTTP **serving
  sidecar** lets any language or service ask for explanations over the network.

Under the hood, the number-crunching lives in a compact **Rust** core (fast, predictable)
with a thin **Python** layer on top for training and everyday use.

---

## Being honest about the limits

SimLens is built to *show* its own uncertainty rather than paper over it:

- **The higher levels are approximate.** Only Level 1 is exact. Levels 2–3 always report a
  residual, and SimLens warns (and suggests dropping to Level 1) when it grows.
- **Naming is the hard part.** Ranking the contributions is exact math; turning "feature
  #2317" into the *right* human label is genuinely difficult, so every label carries a
  confidence score and unnamed-but-important features are shown rather than hidden.
- **A general tool gives up domain priors.** Out of the box on an unfamiliar space the
  concept names will be rough; you make them good by supplying label functions and example
  sets from your own domain.

The guiding rule throughout: an explanation you can't verify is worse than no explanation,
so SimLens always hands you the means to check it.

---

## Try it

The repository ships a runnable, self-contained demo that trains a small dictionary on a
synthetic space with known concepts and walks through every feature above:

```bash
python examples/quickstart.py
```

Start with `simlens.Explainer(metric="cosine")` and a couple of your own vectors — you'll
get a faithful, exact explanation on the very first call, before any training at all.
