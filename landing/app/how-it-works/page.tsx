import type { Metadata } from "next";
import Link from "next/link";
import { CodeBlock, Kw, Str, Cm, Fn, Out } from "@/components/CodeBlock";

export const metadata: Metadata = {
  title: "How SimLens works",
  description:
    "A guided tour of SimLens — how a similarity score decomposes into faithful, named reasons.",
};

const sections = [
  ["problem", "The problem"],
  ["big-idea", "The big idea"],
  ["residual", "The honesty dial"],
  ["levels", "Three levels"],
  ["polarity", "Polarity"],
  ["beyond", "Beyond a single pair"],
  ["bundles", "Bundles"],
  ["anything", "Works with anything"],
  ["limits", "Honest limits"],
  ["try-it", "Try it"],
];

export default function HowItWorks() {
  return (
    <div className="docs-layout">
      <aside className="docs-sidebar">
        <div className="side-group">
          <div className="side-title">On this page</div>
          {sections.map(([id, label]) => (
            <a key={id} href={`#${id}`}>
              {label}
            </a>
          ))}
        </div>
      </aside>

      <article className="prose">
        <h1>How SimLens works</h1>
        <p>
          <em>
            A guided tour for the curious. No prior background in
            interpretability needed — just a rough idea of what an
            &ldquo;embedding&rdquo; is.
          </em>
        </p>

        <h2 id="problem">The problem SimLens solves</h2>
        <p>
          Modern search, recommendation, and retrieval systems turn everything
          — a sentence, an image, a product, a molecule — into a list of
          numbers called an <strong>embedding vector</strong>. Two things are
          considered &ldquo;similar&rdquo; when their vectors point in a
          similar direction. A database compares them and hands you back a
          single number: <code>0.83</code>.
        </p>
        <p>
          That number tells you <em>that</em> two things are similar. It never
          tells you <strong>why</strong>.
        </p>
        <p>
          For a lot of applications that&apos;s fine. But the moment a human
          has to <em>trust</em> or <em>act on</em> the result — a doctor
          looking at &ldquo;these two case reports are related,&rdquo; an
          analyst asking &ldquo;why did the search rank this document
          first,&rdquo; a shopper wondering &ldquo;why was this
          recommended&rdquo; — the bare score is frustrating. You want the
          reasons, in words.
        </p>
        <p>
          <strong>SimLens produces those reasons.</strong> Given the two
          vectors and the similarity metric, it breaks the score down into the
          specific ingredients that produced it, and — crucially — it can{" "}
          <em>prove</em> the breakdown adds up.
        </p>

        <h2 id="big-idea">The big idea: a similarity score is a sum</h2>
        <p>
          Here&apos;s the insight everything rests on. The most common way to
          compare two vectors is the <strong>dot product</strong>: multiply
          them coordinate by coordinate, then add up the results.
        </p>
        <div className="formula">
          score = q₁·c₁ + q₂·c₂ + q₃·c₃ + … + q₇₆₈·c₇₆₈
        </div>
        <p>
          Look at that: the score is <em>literally already a sum</em> of
          per-coordinate pieces. Each term <code>qᵢ·cᵢ</code> is the exact
          amount that coordinate <em>i</em> contributed to the final number.
          Nothing is hidden. If coordinate 412 contributed 0.44 out of a 0.55
          score, then coordinate 412 is 80% of the reason these two things
          matched — and that&apos;s not an estimate, it&apos;s arithmetic.
        </p>
        <p>
          <strong>Cosine similarity</strong> (the other popular metric) is the
          same idea after scaling both vectors to length 1, so it decomposes
          exactly the same way. This gives SimLens its foundation: for the
          metrics almost everyone uses, an <em>exact, honest</em> explanation
          of a similarity score already exists inside the math. SimLens
          surfaces it.
        </p>
        <p>
          This is <strong>Level 1</strong>, and it needs no training, no
          model, no setup — just the two vectors.
        </p>
        <CodeBlock title="level 1 — zero setup">
          <Kw>import</Kw> simlens{"\n"}
          ex = simlens.<Fn>Explainer</Fn>(metric=<Str>"cosine"</Str>){"\n"}
          attr = ex.<Fn>explain</Fn>(query_vec, candidate_vec){"\n"}
          <Fn>print</Fn>(attr.<Fn>as_sentence</Fn>()){"\n"}
          <Out>→ "Matched mainly on 'dim:0' (80%), 'dim:3' (20%)."</Out>
        </CodeBlock>

        <h2 id="residual">The honesty dial: the completeness residual</h2>
        <p>
          Any explanation tool can <em>claim</em> to tell you why. The
          question is whether the explanation is <strong>faithful</strong> —
          whether it actually reflects the math, or is just a plausible story.
        </p>
        <p>
          SimLens attaches a number to every explanation called the{" "}
          <strong>completeness residual</strong>: the gap between the real
          score and the sum of the contributions it reported.
        </p>
        <div className="formula">
          residual = | score − (sum of all contributions) |
        </div>
        <ul>
          <li>
            At Level 1, this residual is <strong>zero</strong> (to
            floating-point precision). The explanation <em>is</em> the
            arithmetic; it cannot be wrong.
          </li>
          <li>
            At the higher levels, the residual can be larger — and SimLens{" "}
            <strong>reports it, every time</strong>, instead of hiding it. If
            the residual gets big, SimLens raises a warning and tells you to
            drop back to the exact level.
          </li>
        </ul>
        <div className="callout good">
          The core design principle: <strong>never present a pretty
          explanation you can&apos;t stand behind.</strong> The residual is
          always in the box.
        </div>

        <h2 id="levels">Zooming in: three levels of explanation</h2>
        <p>
          &ldquo;Coordinate 412 contributed 80%&rdquo; is faithful, but
          coordinate 412 doesn&apos;t <em>mean</em> anything to a human.
          SimLens offers three <strong>zoom levels</strong>, trading a little
          exactness for a lot of readability as you go up.
        </p>

        <h3>Level 1 — Dimensions (exact, zero setup)</h3>
        <p>
          The raw per-coordinate breakdown described above. Always available,
          always exact. Great for debugging and as the ground truth the other
          levels are checked against. Its weakness: individual embedding
          coordinates are usually <strong>polysemantic</strong> — one
          coordinate mixes several unrelated ideas together — so the names
          (&ldquo;dim:412&rdquo;) aren&apos;t meaningful.
        </p>

        <h3>Level 2 — Features (learned, monosemantic)</h3>
        <p>
          To get meaningful units, SimLens learns a <strong>dictionary</strong>{" "}
          of directions in the embedding space using a small neural network
          called a <strong>sparse autoencoder (SAE)</strong>. The SAE is
          trained to reconstruct embeddings using only a handful of
          &ldquo;features&rdquo; active at a time. Because of that sparsity
          pressure, the features it discovers tend to be{" "}
          <strong>monosemantic</strong> — each one lights up for a single,
          coherent concept (&ldquo;this text is about litigation,&rdquo;
          &ldquo;this image contains a face,&rdquo; &ldquo;this molecule has a
          sulfonamide group&rdquo;).
        </p>
        <p>
          SimLens then measures which features <em>both</em> vectors activate,
          and how much each shared feature drove the match:
        </p>
        <div className="formula">
          contribution of feature f = (how strongly q activates f)
          <br />
          {"                         "}× (how strongly c activates f)
          <br />
          {"                         "}× (the feature&apos;s weight in the
          space)
        </div>
        <p>
          Now the explanation reads in terms of features instead of anonymous
          coordinates. And because features are auto-labeled, many of them come
          with human names. Level 2 is an <em>approximation</em> of the score
          — so its residual is usually non-zero, and SimLens shows it honestly.
        </p>

        <h3>Level 3 — Concepts (named, from your examples)</h3>
        <p>
          Sometimes you already know the concepts you care about
          (&ldquo;relevant,&rdquo; &ldquo;toxic,&rdquo; &ldquo;sports
          content,&rdquo; &ldquo;kinase inhibitor&rdquo;). If you can supply a
          handful of positive and negative example vectors for a concept,
          SimLens fits a <strong>concept direction</strong> (a Concept
          Activation Vector, a technique from the interpretability literature)
          and can then decompose any score along your named concepts.
        </p>
        <p>
          Concepts are a <em>chosen subspace</em>, not a complete basis, so
          Level-3 explanations are deliberately <strong>partial</strong> —
          SimLens says so in a warning and reports how much of the score the
          named concepts actually cover.
        </p>
        <p>
          Here&apos;s the same pair of items explained at Levels 2 and 3 (from
          the bundled demo):
        </p>
        <CodeBlock title="levels 2 & 3, side by side">
          feature attribution{"  "}score=0.582{"  "}residual=2.2e-01{"  "}
          coverage=90%{"\n"}
          {"  "}concept_4{"   "}+0.1832 ~0.98{"  "}
          <Out>████████████████████████████</Out>
          {"\n"}
          {"  "}concept_2{"   "}+0.1009 ~0.97{"  "}
          <Out>███████████████</Out>
          {"\n"}
          {"  "}
          <Cm>
            ⚠ completeness_residual_high: Σφ deviates from score by 0.22 (38%);
            use level="dim"…
          </Cm>
          {"\n\n"}
          concept attribution{"  "}score=0.582{"  "}residual=5.1e-02{"  "}
          coverage=99%{"\n"}
          {"  "}concept_4{"   "}+0.3067 ~1.00{"  "}
          <Out>████████████████████████████</Out>
          {"\n"}
          {"  "}concept_2{"   "}+0.2539 ~1.00{"  "}
          <Out>███████████████████████</Out>
          {"\n"}
          {"  "}concept_5{"   "}−0.0322 ~1.00{"  "}
          <Cm>░░░</Cm>
          {"\n"}
          {"  "}
          <Cm>
            ⚠ partial_decomposition: concepts span a subspace, not a complete
            basis…
          </Cm>
        </CodeBlock>
        <p>
          The <code>~0.98</code> is the label&apos;s{" "}
          <strong>confidence</strong>; the bars are signed (solid = pushes
          similarity up, shaded = pushes it down); and the warnings are the
          honesty dial doing its job.
        </p>

        <h2 id="polarity">Shared, or one-sided? (polarity)</h2>
        <p>Every contribution is tagged with a polarity:</p>
        <ul>
          <li>
            <strong>shared</strong> — both items have this feature (it&apos;s
            a <em>reason they match</em>),
          </li>
          <li>
            <strong>query-only</strong> — the query has it but the candidate
            doesn&apos;t,
          </li>
          <li>
            <strong>candidate-only</strong> — the reverse.
          </li>
        </ul>
        <p>
          This lets SimLens answer not just &ldquo;why are these similar&rdquo;
          but the mirror question{" "}
          <strong>
            &ldquo;why aren&apos;t they more similar / why was something
            omitted&rdquo;
          </strong>{" "}
          — you look at the strong query-only features the candidate is
          missing.
        </p>

        <h2 id="beyond">Beyond a single pair</h2>
        <p>
          The additive foundation makes several genuinely useful operations
          fall out for free.
        </p>

        <h3>Why did A outrank B? (margin attribution)</h3>
        <p>
          In search and recommendation, the real question is rarely &ldquo;why
          does A match&rdquo; — it&apos;s{" "}
          <strong>&ldquo;why did A beat B?&rdquo;</strong> Because scores are
          sums, the <em>difference</em> between two scores is also a sum, so
          SimLens can decompose the <strong>margin</strong> directly:
        </p>
        <CodeBlock title="margin attribution">
          ex.<Fn>explain_margin</Fn>(query, better=A, worse=B){"\n"}
          <Out>
            → "Matched mainly on 'concept_4' (52%), 'concept_2' (33%),
            'concept_3' (−15%)."
          </Out>
        </CodeBlock>
        <p>
          Positive terms are why A won; negative terms are places B was
          actually stronger.
        </p>

        <h3>What&apos;s the minimal reason? (ablation)</h3>
        <p>
          SimLens can greedily find the <strong>smallest set of features</strong>{" "}
          whose removal would drop the match below a threshold — the &ldquo;if
          you only remember one thing&rdquo; explanation.
        </p>
        <CodeBlock title="ablation">
          abl = ex.<Fn>ablate</Fn>(query, candidate, threshold=<Out>0.5</Out>)
          {"\n"}
          <Out>→ removing 1 feature: 0.582 → 0.399 (dropped_below=True)</Out>
        </CodeBlock>

        <h3>&ldquo;More like this, but less of that&rdquo; (steering)</h3>
        <p>
          Because concepts are directions, you can <strong>edit a query</strong>{" "}
          in concept space and search again — &ldquo;more like this document
          but less about sports,&rdquo; &ldquo;more recent,&rdquo; &ldquo;less
          lipophilic.&rdquo;
        </p>
        <CodeBlock title="steering">
          new_query = ex.<Fn>steer</Fn>(query, {"{"}
          <Str>"sports"</Str>: <Out>-1.0</Out>, <Str>"finance"</Str>:{" "}
          <Out>+0.5</Out>
          {"}"})
        </CodeBlock>

        <h3>Other views</h3>
        <ul>
          <li>
            <strong>Contrastive-corpus</strong> explanations describe what
            makes a hit similar <em>relative to</em> a background set, which
            sharpens the concepts that get surfaced.
          </li>
          <li>
            <strong>Aspect view</strong> rolls up fine-grained concepts into a
            few big buckets (&ldquo;topic / tone / recency&rdquo;) for an
            at-a-glance summary.
          </li>
          <li>
            <strong>Cohort summary</strong> aggregates over a whole result
            page: &ldquo;8 of these 10 results matched mainly on concept
            X.&rdquo;
          </li>
        </ul>

        <h2 id="bundles">Bundles: portable, auditable artifacts</h2>
        <p>
          The learned dictionary (the SAE) and the named concepts are packaged
          into a <strong>bundle</strong> — a small directory you train once
          and reuse. A bundle is stamped with a <strong>content hash</strong>,
          and every explanation it produces carries that hash. That means an
          explanation is <strong>reproducible and traceable</strong>: you can
          prove which exact artifacts produced a given rationale — useful
          anywhere a decision has to be defensible.
        </p>
        <CodeBlock title="bundles">
          bundle.<Fn>save</Fn>(<Str>"mybundle.simlens"</Str>){"\n"}
          ex = simlens.<Fn>Explainer</Fn>(<Str>"mybundle.simlens"</Str>){"   "}
          <Cm># load and go</Cm>
        </CodeBlock>
        <p>
          Bundles are also how you bring your own dictionary: if you&apos;ve
          trained a sparse autoencoder elsewhere, SimLens can import it
          (safetensors / SAELens via <code>import_safetensors_sae</code>)
          rather than making you retrain.
        </p>

        <h2 id="anything">Works with anything</h2>
        <ul>
          <li>
            <strong>Embedder-agnostic.</strong> It only ever touches the
            vectors, so it works with any embedding model and any modality —
            text, images, audio, structured data, molecules.
          </li>
          <li>
            <strong>Store-agnostic.</strong> It doesn&apos;t replace your
            vector database; it explains its output. Adapters pull candidate
            vectors from common stores, and a small HTTP serving sidecar lets
            any language or service ask for explanations over the network.
          </li>
        </ul>
        <p>
          Under the hood, the number-crunching lives in a compact{" "}
          <strong>Rust</strong> core (fast, predictable) with a thin{" "}
          <strong>Python</strong> layer on top for training and everyday use.
        </p>

        <h2 id="limits">Being honest about the limits</h2>
        <ul>
          <li>
            <strong>The higher levels are approximate.</strong> Only Level 1 is
            exact. Levels 2–3 always report a residual, and SimLens warns (and
            suggests dropping to Level 1) when it grows.
          </li>
          <li>
            <strong>Naming is the hard part.</strong> Ranking the
            contributions is exact math; turning &ldquo;feature #2317&rdquo;
            into the <em>right</em> human label is genuinely difficult, so
            every label carries a confidence score and unnamed-but-important
            features are shown rather than hidden.
          </li>
          <li>
            <strong>A general tool gives up domain priors.</strong> Out of the
            box on an unfamiliar space the concept names will be rough; you
            make them good by supplying label functions and example sets from
            your own domain.
          </li>
        </ul>
        <div className="callout">
          The guiding rule throughout: an explanation you can&apos;t verify is
          worse than no explanation, so SimLens always hands you the means to
          check it.
        </div>

        <h2 id="try-it">Try it</h2>
        <p>
          The repository ships a runnable, self-contained demo that trains a
          small dictionary on a synthetic space with known concepts and walks
          through every feature above:
        </p>
        <CodeBlock title="shell">python examples/quickstart.py</CodeBlock>
        <p>
          Start with <code>simlens.Explainer(metric=&quot;cosine&quot;)</code>{" "}
          and a couple of your own vectors — you&apos;ll get a faithful, exact
          explanation on the very first call, before any training at all.
        </p>
        <p>
          Ready for the details? Head to the{" "}
          <Link href="/docs">full documentation</Link>.
        </p>
      </article>
    </div>
  );
}
