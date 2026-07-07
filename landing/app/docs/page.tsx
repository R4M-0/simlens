import type { Metadata } from "next";
import Link from "next/link";
import { CodeBlock, Kw, Str, Cm, Fn, Out } from "@/components/CodeBlock";

export const metadata: Metadata = {
  title: "SimLens documentation",
  description:
    "Installation, quickstart, capability reference, integrations, adapters, CLI, benchmarks, and validation for SimLens.",
};

const nav: [string, [string, string][]][] = [
  [
    "Start here",
    [
      ["overview", "Overview"],
      ["getting-started", "Installation"],
      ["quickstart", "Quickstart"],
    ],
  ],
  [
    "Core",
    [
      ["capabilities", "Capability reference"],
      ["levels", "Explanation levels"],
      ["centered", "Centered “why”"],
      ["learned-metrics", "Learned metrics (IG)"],
      ["multivector", "Multi-vector"],
    ],
  ],
  [
    "Bundles & training",
    [
      ["bundles", "Bundles"],
      ["autofit", "autofit"],
      ["training", "SAE training"],
      ["naming", "Feature naming"],
      ["certify", "Certification"],
    ],
  ],
  [
    "Ecosystem",
    [
      ["integrations", "Integrations"],
      ["adapters", "Store adapters"],
      ["cli", "CLI & serving"],
    ],
  ],
  [
    "Reference",
    [
      ["benchmarks", "Benchmarks"],
      ["validation", "Validation"],
      ["layout", "Project layout"],
      ["status", "Status & roadmap"],
      ["limits", "Limitations"],
    ],
  ],
];

export default function Docs() {
  return (
    <div className="docs-layout">
      <aside className="docs-sidebar">
        {nav.map(([group, items]) => (
          <div className="side-group" key={group}>
            <div className="side-title">{group}</div>
            {items.map(([id, label]) => (
              <a key={id} href={`#${id}`}>
                {label}
              </a>
            ))}
          </div>
        ))}
      </aside>

      <article className="prose">
        <h1>Documentation</h1>
        <p>
          Everything you need to install, use, and trust SimLens — from a
          zero-setup first call to trained, certified, named concept bundles.
          New to the ideas? Read{" "}
          <Link href="/how-it-works">How SimLens works</Link> first.
        </p>

        {/* ============================ overview ============================ */}
        <h2 id="overview">Overview</h2>
        <p>
          <strong>SimLens</strong> is faithful, vector-only similarity &amp;
          ranking attribution. Given two embeddings and a metric, it
          decomposes the similarity score into additive,
          completeness-checked contributions at three zoom levels — raw
          dimensions (exact), learned SAE features, and named concepts.
        </p>
        <ul>
          <li>
            <strong>Vector-only:</strong> it reads embeddings, never your model
            or your database — so it works with any embedder, any modality,
            any store.
          </li>
          <li>
            <strong>Faithful:</strong> for dot &amp; cosine the contributions
            provably sum to the score; every result carries a{" "}
            <em>completeness residual</em> quantifying any gap.
          </li>
          <li>
            <strong>Fast:</strong> the per-query hot path is a compact Rust
            core (<code>simlens-core</code>) exposed through PyO3 with a
            zero-copy numpy FFI.
          </li>
          <li>
            <strong>Auditable:</strong> bundles are content-hashed and
            signable; every explanation is traceable to the exact artifacts
            that produced it.
          </li>
        </ul>

        {/* ============================ install ============================ */}
        <h2 id="getting-started">Installation</h2>
        <CodeBlock title="pip">
          pip install simlens{"\n\n"}
          <Cm># optional extras (any combination):</Cm>
          {"\n"}
          pip install <Str>"simlens[qdrant,openai,train]"</Str>
        </CodeBlock>
        <div className="table-wrap">
          <table className="doc">
            <thead>
              <tr>
                <th>Extra</th>
                <th>Enables</th>
                <th>Pulls in</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><code>qdrant</code> / <code>pgvector</code> / <code>faiss</code> / <code>weaviate</code></td>
                <td>Vector-store adapters</td>
                <td><code>qdrant-client</code>, <code>psycopg[binary]</code>, <code>faiss-cpu</code>, <code>weaviate-client</code></td>
              </tr>
              <tr>
                <td><code>openai</code> / <code>gemini</code></td>
                <td>LLM feature-naming providers</td>
                <td><code>openai&gt;=1</code>, <code>google-generativeai</code></td>
              </tr>
              <tr>
                <td><code>train</code></td>
                <td>Research-scale GPU SAE training + safetensors import</td>
                <td><code>torch&gt;=2</code>, <code>safetensors</code></td>
              </tr>
              <tr>
                <td><code>kg</code></td>
                <td>Sublinear kNN edge proposal for the KG extension</td>
                <td><code>hnswlib</code></td>
              </tr>
            </tbody>
          </table>
        </div>
        <h3>Build from source</h3>
        <CodeBlock title="shell">
          python -m venv .venv &amp;&amp; . .venv/bin/activate{"\n"}
          pip install maturin numpy pytest{"\n"}
          maturin develop --release{"      "}
          <Cm># builds the Rust core into your environment</Cm>
          {"\n"}
          cargo test -p simlens-core{"     "}
          <Cm># Rust unit tests</Cm>
          {"\n"}
          pytest -q{"                      "}
          <Cm># Python tests</Cm>
          {"\n"}
          python examples/quickstart.py{"  "}
          <Cm># full end-to-end demo</Cm>
        </CodeBlock>
        <p>
          Requires Python ≥ 3.8. Prebuilt abi3 wheels ship on PyPI for Linux
          and macOS (universal2).
        </p>

        {/* ============================ quickstart ============================ */}
        <h2 id="quickstart">Quickstart</h2>
        <p>
          The exact Level-1 explanation needs nothing but two vectors and a
          metric:
        </p>
        <CodeBlock title="zero setup">
          <Kw>import</Kw> simlens{"\n\n"}
          ex = simlens.<Fn>Explainer</Fn>(metric=<Str>"cosine"</Str>){"\n"}
          attr = ex.<Fn>explain</Fn>(query_vec, candidate_vec){"\n"}
          <Fn>print</Fn>(attr.<Fn>as_sentence</Fn>()){"\n"}
          <Out>
            → "Matched mainly on 'financial-regulation' (61%), '2024-filings'
            (22%)."
          </Out>
        </CodeBlock>
        <p>
          To get named, human-readable explanations, point{" "}
          <code>autofit</code> at your store — it samples vectors, trains the
          dictionary, and names features from your payload fields (plus an
          optional LLM), all in one line:
        </p>
        <CodeBlock title="autofit">
          bundle = simlens.<Fn>autofit</Fn>(store){"                       "}
          <Cm># zero manual labeling</Cm>
          {"\n"}
          <Cm>
            # or: simlens.autofit(store,
            namer=simlens.naming.from_provider("openai"))
          </Cm>
        </CodeBlock>
        <p>
          Or walk the full pipeline by hand — the bundled demo does exactly
          this on a synthetic space with known concepts:
        </p>
        <CodeBlock title="examples/quickstart.py (abridged)">
          <Kw>from</Kw> simlens <Kw>import</Kw> train{"\n\n"}
          <Cm># 1. train a sparse autoencoder + auto-label features</Cm>
          {"\n"}
          sae = train.<Fn>SAE</Fn>(dim=dim, expansion=<Out>4</Out>, l1=
          <Out>1e-3</Out>).<Fn>fit</Fn>(X, epochs=<Out>60</Out>){"\n"}
          bundle = train.<Fn>build_bundle</Fn>(<Str>"synthetic-v1"</Str>,{" "}
          <Str>"cosine"</Str>, sae, X, labelers=labelers){"\n\n"}
          <Cm># 2. register named concepts (CAVs) from example vectors</Cm>
          {"\n"}
          bundle.<Fn>add_concept</Fn>(<Str>"finance"</Str>, positives,
          negatives, aspect=<Str>"topic"</Str>){"\n\n"}
          <Cm># 3. save / load / verify (audit)</Cm>
          {"\n"}
          bundle.<Fn>save</Fn>(<Str>"demo.simlens"</Str>){"\n"}
          ex = simlens.<Fn>Explainer</Fn>(simlens.Bundle.<Fn>load</Fn>(
          <Str>"demo.simlens"</Str>)){"\n\n"}
          <Cm># 4. explain at any level</Cm>
          {"\n"}
          ex.<Fn>explain</Fn>(q, c, level=<Str>"feature"</Str>, top_k=
          <Out>5</Out>){"\n"}
          ex.<Fn>explain</Fn>(q, c, level=<Str>"concept"</Str>){"\n"}
          ex.<Fn>explain_margin</Fn>(q, better, worse, level=
          <Str>"concept"</Str>)
        </CodeBlock>

        {/* ============================ capabilities ============================ */}
        <h2 id="capabilities">Capability reference</h2>
        <div className="table-wrap">
          <table className="doc">
            <thead>
              <tr>
                <th>Capability</th>
                <th>Call</th>
              </tr>
            </thead>
            <tbody>
              <tr><td>Explain a match (dims / features / concepts / aspects)</td><td><code>ex.explain(q, c, level=...)</code></td></tr>
              <tr><td>Explain a <strong>ranking</strong> — why A beat B</td><td><code>ex.explain_margin(q, better, worse)</code></td></tr>
              <tr><td><strong>Centered “why”</strong> — discriminative, past the anisotropy baseline</td><td><code>ex.explain(q, c, center=True)</code></td></tr>
              <tr><td><strong>Any learned metric</strong> (cross-encoder, reranker) via Integrated Gradients</td><td><code>simlens.LearnedMetricExplainer(scorer).explain(q, c)</code></td></tr>
              <tr><td>Why they&apos;re <strong>not</strong> more similar</td><td><code>ex.explain_dissimilarity(q, c)</code></td></tr>
              <tr><td>Minimal reason — what breaks the match</td><td><code>ex.ablate(q, c, threshold=...)</code></td></tr>
              <tr><td>Steer a query in concept space</td><td><code>ex.steer(q, {"{"}&quot;topic&quot;: -1.0{"}"})</code></td></tr>
              <tr><td>Contrast against a background set</td><td><code>ex.explain_vs_corpus(q, c, foil)</code></td></tr>
              <tr><td><strong>Certify</strong> a bundle&apos;s faithfulness (signed scorecard)</td><td><code>bundle.certify(vectors)</code></td></tr>
              <tr><td>Summarize a whole result page</td><td><code>ex.summarize(q, hits)</code></td></tr>
              <tr><td>Late-interaction (multi-vector) attribution</td><td><code>simlens.MultiVectorExplainer().explain(Q, C)</code></td></tr>
              <tr><td><strong>Auto-build a bundle from your store</strong></td><td><code>simlens.autofit(store)</code></td></tr>
              <tr><td>Train &amp; package a concept bundle manually</td><td><code>simlens.train.build_bundle(...)</code></td></tr>
              <tr><td>Calibrate / audit naming confidence</td><td><code>simlens.eval.reliability(bundle, X, labelers)</code></td></tr>
              <tr><td>Fetch candidates from your store</td><td><code>simlens.adapters.Qdrant / Pgvector / Faiss / Weaviate</code></td></tr>
              <tr><td>Inspect, verify, evaluate from the shell</td><td><code>simlens info | verify | eval | serve</code></td></tr>
              <tr><td>Serve over HTTP</td><td><code>python -m simlens.serve --bundle b.simlens</code></td></tr>
            </tbody>
          </table>
        </div>

        {/* ============================ levels ============================ */}
        <h2 id="levels">Explanation levels</h2>
        <p>
          Pass <code>level=</code> to <code>ex.explain(...)</code>. All levels
          share one additive contract: contributions (each tagged{" "}
          <code>shared</code> / <code>query-only</code> /{" "}
          <code>candidate-only</code>) plus a completeness residual.
        </p>
        <div className="table-wrap">
          <table className="doc">
            <thead>
              <tr>
                <th>Level</th>
                <th>Units</th>
                <th>Setup</th>
                <th>Exactness</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><code>&quot;dim&quot;</code></td>
                <td>Raw embedding coordinates</td>
                <td>None</td>
                <td>Exact — residual 0 to float precision</td>
              </tr>
              <tr>
                <td><code>&quot;feature&quot;</code></td>
                <td>Learned monosemantic SAE features (auto-named)</td>
                <td>Bundle (train or <code>autofit</code>)</td>
                <td>Approximate — residual = SAE reconstruction error, always reported</td>
              </tr>
              <tr>
                <td><code>&quot;concept&quot;</code></td>
                <td>Your named concepts (CAVs from example vectors)</td>
                <td>Bundle + <code>add_concept</code></td>
                <td>Deliberately partial — coverage reported, warning attached</td>
              </tr>
              <tr>
                <td><code>&quot;aspect&quot;</code></td>
                <td>Concepts rolled up into buckets (topic / tone / recency…)</td>
                <td>Concepts with <code>aspect=</code> tags</td>
                <td>Same contract as concepts, coarser view</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div className="callout warn">
          When the residual grows past its threshold, the attribution carries a{" "}
          <code>completeness_residual_high</code> warning telling you to drop
          to <code>level=&quot;dim&quot;</code> — SimLens never silently
          presents an approximation as exact.
        </div>

        {/* ============================ centered ============================ */}
        <h2 id="centered">Centered “why” (anisotropy correction)</h2>
        <p>
          Real embedding spaces are <em>anisotropic</em>: all vectors share a
          large common component, so a raw decomposition is dominated by the
          shared-mean baseline (&ldquo;everything matches on
          &lsquo;being English text&rsquo;&rdquo;). Centering removes that
          baseline and surfaces the <strong>discriminative</strong> concepts —
          the reasons <em>this</em> pair matched and others didn&apos;t.
        </p>
        <CodeBlock title="centering">
          <Cm># per-call:</Cm>
          {"\n"}
          ex.<Fn>explain</Fn>(q, c, center=<Kw>True</Kw>){"\n\n"}
          <Cm>
            # or fit a correction (mean / ABTT / whitening) from a sample:
          </Cm>
          {"\n"}
          ctr = simlens.<Fn>fit_centering</Fn>(X, method=<Str>"abtt"</Str>)
          {"\n"}
          simlens.<Fn>apply_centering</Fn>(ctr, vecs)
        </CodeBlock>
        <p>
          Centering is the default in the integrations — on real embeddings
          it&apos;s what makes the surfaced topic actually discriminative.
        </p>

        {/* ============================ learned metrics ============================ */}
        <h2 id="learned-metrics">Learned metrics — Integrated Gradients</h2>
        <p>
          Dot and cosine decompose exactly, but rerankers and cross-encoders
          don&apos;t. For any learned or non-linear scorer,{" "}
          <code>LearnedMetricExplainer</code> uses{" "}
          <strong>Integrated Gradients</strong>, which comes with a
          completeness axiom — attributions sum to the difference between the
          score and a baseline, and the residual is still reported.
        </p>
        <CodeBlock title="learned metric">
          lex = simlens.<Fn>LearnedMetricExplainer</Fn>(scorer){"   "}
          <Cm># scorer: (q, c) → score</Cm>
          {"\n"}
          attr = lex.<Fn>explain</Fn>(q, c)
        </CodeBlock>

        {/* ============================ multivector ============================ */}
        <h2 id="multivector">Multi-vector (late interaction)</h2>
        <p>
          For ColBERT-style late-interaction models, where a query and a
          document are each a <em>set</em> of vectors,{" "}
          <code>MultiVectorExplainer</code> attributes the score over
          token-level vector pairs:
        </p>
        <CodeBlock title="multi-vector">
          attr = simlens.<Fn>MultiVectorExplainer</Fn>().<Fn>explain</Fn>(Q, C)
        </CodeBlock>

        {/* ============================ bundles ============================ */}
        <h2 id="bundles">Bundles</h2>
        <p>
          A <strong>bundle</strong> packages the learned dictionary (SAE), the
          feature names, and the named concepts into a small directory you
          train once and reuse. It is stamped with a{" "}
          <strong>content hash</strong>; every explanation it produces carries
          that hash, so any rationale is reproducible and traceable to the
          exact artifacts that made it.
        </p>
        <CodeBlock title="save / load / verify">
          bundle.<Fn>save</Fn>(<Str>"mybundle.simlens"</Str>){"\n"}
          b = simlens.Bundle.<Fn>load</Fn>(<Str>"mybundle.simlens"</Str>){"\n"}
          b.<Fn>verify</Fn>(){"          "}
          <Cm># hash (and signature) check</Cm>
          {"\n"}
          ex = simlens.<Fn>Explainer</Fn>(b)
        </CodeBlock>
        <p>
          You can also bring your own dictionary: SAEs trained elsewhere drop
          in via <code>import_safetensors_sae</code> (safetensors / SAELens
          format) instead of retraining.
        </p>

        {/* ============================ autofit ============================ */}
        <h2 id="autofit">autofit — store to bundle in one line</h2>
        <p>
          <code>simlens.autofit(store)</code> samples vectors from a store
          adapter, trains the dictionary, and names features from your payload
          fields — optionally with an LLM naming provider:
        </p>
        <CodeBlock title="autofit">
          bundle = simlens.<Fn>autofit</Fn>(store){"\n"}
          bundle = simlens.<Fn>autofit</Fn>(store, namer=simlens.naming.
          <Fn>from_provider</Fn>(<Str>"openai"</Str>))
        </CodeBlock>

        {/* ============================ training ============================ */}
        <h2 id="training">SAE training</h2>
        <p>
          <code>simlens.train</code> ships real sparse-autoencoder trainers
          with trustworthy defaults:
        </p>
        <ul>
          <li>
            <strong>Architectures:</strong> TopK (default), BatchTopK, and
            JumpReLU — with AuxK dead-feature revival.
          </li>
          <li>
            <strong>Backends:</strong> a zero-dependency numpy backend, plus
            an optional <code>torch</code>/GPU backend (extra:{" "}
            <code>train</code>) for research scale.
          </li>
          <li>
            <strong>Train == inference:</strong> the training-time sparsity
            gate is reproduced exactly at inference, so the residual you
            certify is the residual you serve.
          </li>
        </ul>
        <CodeBlock title="training + packaging">
          <Kw>from</Kw> simlens <Kw>import</Kw> train{"\n"}
          sae = train.<Fn>SAE</Fn>(dim=<Out>768</Out>, expansion=<Out>8</Out>).
          <Fn>fit</Fn>(X, epochs=<Out>60</Out>){"\n"}
          bundle = train.<Fn>build_bundle</Fn>(<Str>"prod-v1"</Str>,{" "}
          <Str>"cosine"</Str>, sae, X, labelers=labelers)
        </CodeBlock>

        {/* ============================ naming ============================ */}
        <h2 id="naming">Feature naming</h2>
        <p>
          Ranking contributions is exact math; naming them is the hard part —
          so SimLens <em>measures</em> its names instead of asserting them.
          Names carry a <strong>balanced-accuracy score</strong> (the name
          used as a classifier over held-out activations), and low-scoring
          names are dropped rather than shown as confident nonsense.
          Unnamed-but-important features are shown, not hidden.
        </p>
        <ul>
          <li>
            <strong>Labelers:</strong> supply label functions / label arrays
            over your corpus and <code>build_bundle</code> auto-names features
            against them.
          </li>
          <li>
            <strong>LLM providers:</strong>{" "}
            <code>simlens.naming.from_provider(&quot;openai&quot;)</code> or{" "}
            <code>&quot;gemini&quot;</code> name features from top-activating
            payload examples.
          </li>
          <li>
            <strong>Reliability audit:</strong>{" "}
            <code>simlens.eval.reliability(bundle, X, labelers)</code>{" "}
            calibrates and audits naming confidence.
          </li>
        </ul>

        {/* ============================ certify ============================ */}
        <h2 id="certify">Faithfulness certification</h2>
        <p>
          <code>bundle.certify(vectors)</code> computes a signed quality
          scorecard baked into the bundle and covered by its content hash:
        </p>
        <ul>
          <li><strong>FVU</strong> — fraction of variance unexplained by the SAE,</li>
          <li><strong>L0</strong> — average active features per vector,</li>
          <li><strong>dead %</strong> — features that never fire,</li>
          <li><strong>deletion / insertion AUC</strong> — do the top attributions actually move the score?</li>
          <li><strong>detection accuracies</strong> — how well the names classify.</li>
        </ul>
        <p>
          <code>simlens info &lt;bundle&gt;</code> prints the scorecard, so
          regressions between bundle versions are visible at a glance.
        </p>

        {/* ============================ integrations ============================ */}
        <h2 id="integrations">Integrations (system extensions)</h2>
        <p>
          Thin, customizable wrappers that adapt SimLens to a system type —
          you bring the vectors, they bring the explanation. Centered
          &ldquo;why&rdquo; is the default. Each ships a runnable notebook (
          <code>examples/notebook_*.py</code>) validated on a real embedder —
          see <a href="#validation">Validation</a>.
        </p>
        <CodeBlock title="imports">
          <Kw>from</Kw> simlens.integrations.rag{"    "}
          <Kw>import</Kw> RagExplainer{"           "}
          <Cm># why retrieved / why ranked</Cm>
          {"\n"}
          <Kw>from</Kw> simlens.integrations.recsys <Kw>import</Kw>{" "}
          RecsysExplainer{"        "}
          <Cm># "because you liked …", steer</Cm>
          {"\n"}
          <Kw>from</Kw> simlens.integrations.kg{"     "}
          <Kw>import</Kw> KnowledgeGraphExplainer{" "}
          <Cm># explain / propose / type edges</Cm>
          {"\n"}
          <Kw>from</Kw> simlens.integrations.audit{"  "}
          <Kw>import</Kw> AuditLog{"               "}
          <Cm># signed, hashed decision records</Cm>
        </CodeBlock>
        <div className="table-wrap">
          <table className="doc">
            <thead>
              <tr>
                <th>Integration</th>
                <th>What it answers</th>
                <th>Notebook</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>RAG</strong></td>
                <td>Why was this chunk retrieved / ranked here? Flags spurious features.</td>
                <td><code>examples/notebook_rag.py</code></td>
              </tr>
              <tr>
                <td><strong>Recsys</strong></td>
                <td>A faithful “because you liked …”, plus concept-space steering.</td>
                <td><code>examples/notebook_recsys.py</code></td>
              </tr>
              <tr>
                <td><strong>Knowledge graph</strong></td>
                <td>Explain, propose (kNN, sublinear with <code>[kg]</code>), and type edges.</td>
                <td><code>examples/notebook_kg.py</code></td>
              </tr>
              <tr>
                <td><strong>Audit</strong></td>
                <td>Signed, hashed decision records; deterministic re-explanation.</td>
                <td><code>examples/notebook_audit.py</code></td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* ============================ adapters ============================ */}
        <h2 id="adapters">Store adapters</h2>
        <p>
          SimLens doesn&apos;t replace your vector database — it explains its
          output. Adapters pull candidate vectors (and payloads, for naming)
          from common stores behind one interface:
        </p>
        <div className="table-wrap">
          <table className="doc">
            <thead>
              <tr>
                <th>Adapter</th>
                <th>Import</th>
                <th>Extra</th>
              </tr>
            </thead>
            <tbody>
              <tr><td>Qdrant</td><td><code>simlens.adapters.Qdrant</code></td><td><code>simlens[qdrant]</code></td></tr>
              <tr><td>pgvector</td><td><code>simlens.adapters.Pgvector</code></td><td><code>simlens[pgvector]</code></td></tr>
              <tr><td>FAISS</td><td><code>simlens.adapters.Faiss</code></td><td><code>simlens[faiss]</code></td></tr>
              <tr><td>Weaviate</td><td><code>simlens.adapters.Weaviate</code></td><td><code>simlens[weaviate]</code></td></tr>
              <tr><td>In-memory</td><td><code>simlens.adapters.Memory</code></td><td>— (built in; great for tests)</td></tr>
            </tbody>
          </table>
        </div>
        <p>
          Swapping <code>Memory</code> for <code>Qdrant</code> or{" "}
          <code>Pgvector</code> leaves integration code unchanged.
        </p>

        {/* ============================ cli ============================ */}
        <h2 id="cli">CLI &amp; serving</h2>
        <p>
          The <code>simlens</code> command inspects, verifies, and evaluates
          bundles from the shell; a small HTTP sidecar serves explanations to
          any language or service.
        </p>
        <div className="table-wrap">
          <table className="doc">
            <thead>
              <tr>
                <th>Command</th>
                <th>What it does</th>
              </tr>
            </thead>
            <tbody>
              <tr><td><code>simlens info &lt;bundle&gt;</code></td><td>Print a bundle&apos;s manifest (including the certification scorecard)</td></tr>
              <tr><td><code>simlens verify &lt;bundle&gt;</code></td><td>Verify a bundle&apos;s content hash (and signature)</td></tr>
              <tr><td><code>simlens eval &lt;bundle&gt; &lt;vectors.npy&gt;</code></td><td>Faithfulness scorecard over an <code>[N, dim]</code> array</td></tr>
              <tr><td><code>simlens serve</code></td><td>Start the HTTP serving sidecar</td></tr>
            </tbody>
          </table>
        </div>
        <CodeBlock title="serving sidecar">
          python -m simlens.serve --bundle b.simlens
        </CodeBlock>

        {/* ============================ benchmarks ============================ */}
        <h2 id="benchmarks">Benchmarks</h2>
        <p>
          Generated by <code>python benchmarks/bench.py --md</code>. Times are
          per-call medians on the host CPU; treat them as relative, not
          absolute. <code>explain dim</code> is the exact Level-1
          decomposition (no bundle); <code>explain feature</code> is Level-2
          over a TopK SAE (expansion 8, k=32); encode paths use the zero-copy
          numpy FFI.
        </p>
        <div className="table-wrap">
          <table className="doc">
            <thead>
              <tr>
                <th>dim</th>
                <th>features</th>
                <th>explain dim (µs)</th>
                <th>explain feature (µs)</th>
                <th>SAE encode (µs)</th>
                <th>batch encode (vec/s)</th>
                <th>explain peak (KB)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="num">384</td><td className="num">3,072</td>
                <td className="num">207.23</td><td className="num">3,620.44</td>
                <td className="num">1,630.19</td><td className="num">551</td>
                <td className="num">29.0</td>
              </tr>
              <tr>
                <td className="num">768</td><td className="num">6,144</td>
                <td className="num">257.07</td><td className="num">10,386.83</td>
                <td className="num">5,127.19</td><td className="num">199</td>
                <td className="num">52.6</td>
              </tr>
              <tr>
                <td className="num">1,536</td><td className="num">12,288</td>
                <td className="num">345.37</td><td className="num">42,897.14</td>
                <td className="num">20,976.45</td><td className="num">31</td>
                <td className="num">100.8</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p>
          Run <code>cargo bench -p simlens-core</code> for the Rust-kernel
          micro-benchmarks.
        </p>

        {/* ============================ validation ============================ */}
        <h2 id="validation">Real-system validation</h2>
        <p>
          Each integration ships a runnable notebook that runs on a real
          embedder (fastembed / <code>bge-small-en-v1.5</code>) when available
          and falls back to an offline hashing embedder so it runs anywhere —
          including CI, where <code>tests/test_notebooks.py</code> executes
          all four end-to-end as a guard. Every notebook reports a qualitative
          explanation and one quantitative faithfulness signal:
        </p>
        <div className="table-wrap">
          <table className="doc">
            <thead>
              <tr>
                <th>Integration</th>
                <th>Quantitative signal</th>
                <th>Result</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>RAG</strong></td>
                <td>deletion-AUC of top dims vs. random</td>
                <td><code>top 14.8 &lt; random 18.9</code> → <strong>faithful</strong></td>
              </tr>
              <tr>
                <td><strong>Recsys</strong></td>
                <td>margin reconstructs the score gap</td>
                <td><code>residual 0.0</code> → <strong>exact</strong></td>
              </tr>
              <tr>
                <td><strong>KG</strong></td>
                <td>same-topic edge coherence vs. random</td>
                <td><code>0.87 vs 0.25</code> → <strong>coherent</strong></td>
              </tr>
              <tr>
                <td><strong>Audit</strong></td>
                <td>signatures verify + deterministic re-explain</td>
                <td><strong>reproducible</strong></td>
              </tr>
            </tbody>
          </table>
        </div>
        <h3>What the results say</h3>
        <ul>
          <li>
            <strong>Faithful attribution transfers to real embeddings.</strong>{" "}
            Level-1 (dim) and margin decompositions are exact (residual 0);
            Level-2 residual is just SAE reconstruction error and is small on
            similar pairs. Deletion curves beat random on real vectors.
          </li>
          <li>
            <strong>The centered “why” earns its keep.</strong> On anisotropic
            real embeddings the raw feature decomposition is dominated by the
            shared-mean baseline; the centered view surfaces the
            discriminative topic instead.
          </li>
          <li>
            <strong>The concept level stays honestly partial.</strong> Concept
            attribution reports a nonzero completeness residual by
            construction — the ranking is right, the decomposition is a
            subspace projection, and the warning says so. That&apos;s a{" "}
            <em>finding</em>, not a failure.
          </li>
        </ul>
        <div className="callout">
          Treat a weak signal as information: a high feature-level residual
          means the SAE under-trained for that space (retrain with more{" "}
          <code>epochs</code>/<code>expansion</code>, or import a downloaded
          bundle); low KG coherence means the concepts aren&apos;t
          discriminative there. The <code>Bundle.certify</code> scorecard
          records these numbers so regressions stay visible.
        </div>

        {/* ============================ layout ============================ */}
        <h2 id="layout">Project layout</h2>
        <div className="table-wrap">
          <table className="doc">
            <thead>
              <tr>
                <th>Path</th>
                <th>What</th>
              </tr>
            </thead>
            <tbody>
              <tr><td><code>crates/simlens-core</code></td><td>Rust attribution kernels — the math and the hot path</td></tr>
              <tr><td><code>crates/simlens-py</code></td><td>PyO3 bindings → <code>simlens._native</code></td></tr>
              <tr><td><code>python/simlens</code></td><td>Python API: <code>Explainer</code>, <code>Bundle</code>, <code>train</code>, <code>eval</code>, <code>adapters</code>, <code>viz</code>, <code>serve</code></td></tr>
              <tr><td><code>examples/</code></td><td>Runnable, self-contained demos and validation notebooks</td></tr>
              <tr><td><code>docs/</code></td><td>How SimLens works · benchmarks · validation</td></tr>
            </tbody>
          </table>
        </div>

        {/* ============================ status ============================ */}
        <h2 id="status">Status &amp; roadmap</h2>
        <p>
          <strong>v0.2 — trustworthy defaults, measured not asserted.</strong>{" "}
          Everything in v0.1 plus:
        </p>
        <ul>
          <li>
            <strong>Real SAE trainers</strong> — TopK (default), BatchTopK,
            JumpReLU with AuxK dead-feature revival; numpy and torch/GPU
            backends; train == inference sparsity; safetensors/SAELens import.
          </li>
          <li>
            <strong>Detection-scored naming</strong> — names carry a measured
            balanced accuracy; low-scoring names are dropped.
          </li>
          <li>
            <strong>Centered “why”</strong> — anisotropy correction (mean /
            ABTT / whitening); the default in the integrations.
          </li>
          <li>
            <strong>Integrated Gradients</strong> — explain any
            learned/non-linear scorer with a completeness axiom.
          </li>
          <li>
            <strong>Faithfulness certification</strong> — a signed quality
            scorecard baked into the bundle and covered by its content hash.
          </li>
          <li>
            <strong>Zero-copy numpy FFI</strong>, criterion + Python
            benchmarks, CI + abi3 wheels + PyPI publishing, property-based
            &amp; adversarial tests, and per-extension real-system validation.
          </li>
        </ul>
        <p>
          <strong>Frontier (post-v0.2):</strong> a native Rust{" "}
          <code>simlens-serve</code>, cross-modal and hierarchical concepts.
        </p>

        {/* ============================ limits ============================ */}
        <h2 id="limits">Limitations</h2>
        <ul>
          <li>
            <strong>Only Level 1 is exact.</strong> Levels 2–3 always report a
            residual and warn when it grows — heed the warning.
          </li>
          <li>
            <strong>Naming is genuinely hard.</strong> Labels carry measured
            confidence; unnamed-but-important features are surfaced, not
            hidden.
          </li>
          <li>
            <strong>No domain priors out of the box.</strong> On an unfamiliar
            space, supply label functions and example sets from your own
            domain to make the names good.
          </li>
        </ul>
        <p>
          SimLens is released under the{" "}
          <a
            href="https://github.com/ghassenov/simlens/blob/main/LICENSE"
            target="_blank"
            rel="noreferrer"
          >
            Apache License 2.0
          </a>
          .
        </p>
      </article>
    </div>
  );
}
