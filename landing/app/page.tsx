import Link from "next/link";
import { CodeBlock, Kw, Str, Cm, Fn, Out } from "@/components/CodeBlock";

export default function Home() {
  return (
    <main>
      {/* ---------------- hero ---------------- */}
      <section className="hero">
        <div className="container">
          <img className="logo" src="/logo.svg" alt="SimLens" />
          <h1>
            See <em>why</em> your vectors match.
          </h1>
          <p className="sub">
            Faithful, vector-only similarity &amp; ranking attribution — for
            any embedder, any vector store. Think of it as{" "}
            <strong>SHAP for vector search</strong>: the missing explanation
            layer for similarity and ranking.
          </p>
          <div className="cta-row">
            <Link href="/docs#getting-started" className="btn btn-primary">
              Get started
            </Link>
            <Link href="/how-it-works" className="btn btn-ghost">
              How it works
            </Link>
          </div>
          <div className="badge-row">
            <span className="badge">Apache-2.0</span>
            <span className="badge">core: Rust</span>
            <span className="badge">python ≥ 3.8</span>
            <span className="badge">v0.2</span>
          </div>

          <div className="hero-demo">
            <CodeBlock title="python — zero setup, exact on the first call">
              <Kw>import</Kw> simlens{"\n\n"}
              <Cm>
                # Zero setup — an exact explanation on the very first call, no
                training required.
              </Cm>
              {"\n"}
              ex = simlens.<Fn>Explainer</Fn>(metric=<Str>"cosine"</Str>){"\n"}
              attr = ex.<Fn>explain</Fn>(query_vec, candidate_vec){"\n\n"}
              <Fn>print</Fn>(attr.<Fn>as_sentence</Fn>()){"\n"}
              <Out>
                → "Matched mainly on 'financial-regulation' (61%),{"\n"}
                {"   "}'2024-filings' (22%)."
              </Out>
            </CodeBlock>
          </div>
        </div>
      </section>

      {/* ---------------- problem ---------------- */}
      <section className="block alt">
        <div className="container">
          <div className="kicker">The problem</div>
          <h2 className="title">
            Vector search answers every question with a black box
          </h2>
          <p className="lede">
            Vector search powers your RAG pipeline, your recommendations, your
            semantic search, your anomaly detection. And every one of them
            answers a question with: <em>this matches — score 0.83</em>. Which
            concepts drove the match? Why did the wrong result rank first? Why
            was the right one buried at #7? The score won&apos;t say.
          </p>
          <p className="lede" style={{ marginTop: 16 }}>
            <strong style={{ color: "var(--text)" }}>
              SimLens turns that score into an answer
            </strong>{" "}
            — the specific, named concepts that produced it, in plain language,
            and it <em>proves</em> the breakdown adds up. It reads only the
            vectors, so it drops into any stack without touching your model or
            your database.
          </p>
        </div>
      </section>

      {/* ---------------- features ---------------- */}
      <section className="block" id="features">
        <div className="container">
          <div className="kicker">Why teams reach for SimLens</div>
          <h2 className="title">Faithful by construction, not by storytelling</h2>
          <div className="grid cols-3">
            <div className="card">
              <h3>
                <span className="icon">🎯</span> Answers the real questions
              </h3>
              <p>
                Not just <em>why does A match?</em> but{" "}
                <strong>why did A outrank B?</strong> — the margin, decomposed.
                The question search and recsys teams actually debug.
              </p>
            </div>
            <div className="card">
              <h3>
                <span className="icon">🧾</span> Faithful by construction
              </h3>
              <p>
                For dot &amp; cosine, the explanation <em>is</em> the
                arithmetic: contributions provably sum to the score. Every
                result ships a <strong>completeness residual</strong> so you
                can see exactly how exact it is.
              </p>
            </div>
            <div className="card">
              <h3>
                <span className="icon">🔍</span> Three levels of zoom
              </h3>
              <p>
                Raw dimensions (exact) → learned{" "}
                <strong>monosemantic features</strong> → your own{" "}
                <strong>named concepts</strong> — one consistent, additive
                contract across all three.
              </p>
            </div>
            <div className="card">
              <h3>
                <span className="icon">🎛️</span> Steerable
              </h3>
              <p>
                &ldquo;More like this, but less of <em>that</em>&rdquo; — edit
                a query in concept space with <code>ex.steer()</code> and
                search again.
              </p>
            </div>
            <div className="card">
              <h3>
                <span className="icon">⚡</span> Drop-in &amp; fast
              </h3>
              <p>
                Embedder-agnostic, database-agnostic, and backed by a compact
                Rust core for the per-query hot path. Zero-copy numpy FFI.
              </p>
            </div>
            <div className="card">
              <h3>
                <span className="icon">🔏</span> Auditable
              </h3>
              <p>
                Every explanation is stamped with a content hash of the
                artifacts that produced it — reproducible rationale for
                decisions that have to hold up.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ---------------- levels ---------------- */}
      <section className="block alt">
        <div className="container">
          <div className="kicker">Three levels of explanation</div>
          <h2 className="title">Zoom from exact math to named concepts</h2>
          <p className="lede">
            &ldquo;Coordinate 412 contributed 80%&rdquo; is faithful, but it
            doesn&apos;t <em>mean</em> anything to a human. SimLens trades a
            little exactness for a lot of readability as you zoom — and always
            reports the residual so you know the trade you made.
          </p>
          <div className="levels">
            <div className="level l1">
              <span className="lv-tag">Level 1 · exact</span>
              <h3>Dimensions</h3>
              <p>
                The raw per-coordinate breakdown. Always available, always
                exact, zero setup — the ground truth the other levels are
                checked against.
              </p>
              <div className="residual">residual = 0 (float precision)</div>
            </div>
            <div className="level l2">
              <span className="lv-tag">Level 2 · learned</span>
              <h3>Features</h3>
              <p>
                A sparse autoencoder learns a dictionary of monosemantic
                directions — each feature lights up for a single, coherent
                concept, auto-labeled with measured confidence.
              </p>
              <div className="residual">residual reported, every time</div>
            </div>
            <div className="level l3">
              <span className="lv-tag">Level 3 · named</span>
              <h3>Concepts</h3>
              <p>
                Bring a handful of positive/negative examples per concept and
                SimLens fits concept directions (CAVs) to decompose scores
                along <em>your</em> vocabulary.
              </p>
              <div className="residual">honestly partial, coverage shown</div>
            </div>
          </div>
        </div>
      </section>

      {/* ---------------- capabilities ---------------- */}
      <section className="block">
        <div className="container">
          <div className="kicker">Beyond a single pair</div>
          <h2 className="title">What you can do</h2>
          <div className="grid cols-2">
            <CodeBlock title="why did A outrank B? — margin attribution">
              ex.<Fn>explain_margin</Fn>(query, better=A, worse=B){"\n"}
              <Out>
                → "Matched mainly on 'concept_4' (52%),{"\n"}
                {"   "}'concept_2' (33%), 'concept_3' (−15%)."
              </Out>
            </CodeBlock>
            <CodeBlock title="minimal reason — ablation">
              abl = ex.<Fn>ablate</Fn>(query, candidate, threshold=
              <Out>0.5</Out>){"\n"}
              <Out>
                → removing 1 feature: 0.582 → 0.399{"\n"}
                {"  "}(dropped_below=True)
              </Out>
            </CodeBlock>
            <CodeBlock title="steer — more like this, less of that">
              new_query = ex.<Fn>steer</Fn>(query,{"\n"}
              {"    "}
              {"{"}
              <Str>"sports"</Str>: <Out>-1.0</Out>, <Str>"finance"</Str>:{" "}
              <Out>+0.5</Out>
              {"}"})
            </CodeBlock>
            <CodeBlock title="autofit — one line from store to bundle">
              bundle = simlens.<Fn>autofit</Fn>(store){"\n"}
              <Cm>
                # samples vectors, trains the dictionary,{"\n"}# names features
                from payload fields + LLM
              </Cm>
            </CodeBlock>
          </div>
          <div style={{ textAlign: "center", marginTop: 36 }}>
            <Link href="/docs#capabilities" className="btn btn-ghost">
              See the full capability table →
            </Link>
          </div>
        </div>
      </section>

      {/* ---------------- integrations ---------------- */}
      <section className="block alt">
        <div className="container">
          <div className="kicker">System extensions</div>
          <h2 className="title">
            Thin wrappers for the systems you already run
          </h2>
          <p className="lede">
            You bring the vectors, they bring the explanation.
            Business-logic-agnostic, configurable, and validated on real
            embedders.
          </p>
          <div className="grid cols-2">
            <div className="card">
              <h3>
                <span className="icon">📚</span> RAG
              </h3>
              <p>
                <code>simlens.integrations.rag.RagExplainer</code> — why
                retrieved / why ranked, with spurious-feature flags. Validated:
                deletion-AUC of top dims beats random (14.8 &lt; 18.9).
              </p>
            </div>
            <div className="card">
              <h3>
                <span className="icon">🎬</span> Recommendations
              </h3>
              <p>
                <code>simlens.integrations.recsys.RecsysExplainer</code> — a
                faithful &ldquo;because you liked…&rdquo;, plus steering.
                Margin reconstructs the score gap exactly (residual 0.0).
              </p>
            </div>
            <div className="card">
              <h3>
                <span className="icon">🕸️</span> Knowledge graphs
              </h3>
              <p>
                <code>simlens.integrations.kg.KnowledgeGraphExplainer</code> —
                explain, propose, and type edges. Same-topic edge coherence
                0.87 vs 0.25 random.
              </p>
            </div>
            <div className="card">
              <h3>
                <span className="icon">🧾</span> Audit
              </h3>
              <p>
                <code>simlens.integrations.audit.AuditLog</code> — signed,
                hashed decision records; signatures verify and re-explanation
                is deterministic.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ---------------- install ---------------- */}
      <section className="block">
        <div className="container" style={{ textAlign: "center" }}>
          <div className="kicker">Get started</div>
          <h2 className="title">One pip install away from your first “why”</h2>
          <div className="hero-demo" style={{ marginTop: 28 }}>
            <CodeBlock title="shell">
              pip install simlens{"\n\n"}
              <Cm># extras — stores, LLM naming, GPU training:</Cm>
              {"\n"}
              pip install <Str>"simlens[qdrant,openai,train]"</Str>
            </CodeBlock>
          </div>
          <div className="cta-row">
            <Link href="/docs" className="btn btn-primary">
              Read the docs
            </Link>
            <a
              className="btn btn-ghost"
              href="https://github.com/ghassenov/simlens"
              target="_blank"
              rel="noreferrer"
            >
              Star on GitHub
            </a>
          </div>
        </div>
      </section>
    </main>
  );
}
