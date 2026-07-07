# SimLens landing site

The marketing + documentation website for [SimLens](https://github.com/ghassenov/simlens),
built with Next.js (App Router, static export).

## Pages

- `/` — landing: problem, features, explanation levels, capabilities, integrations, install
- `/how-it-works` — the guided tour (adapted from `docs/how-simlens-works.md`)
- `/docs` — detailed documentation: install, quickstart, capability reference, bundles,
  training, naming, certification, integrations, adapters, CLI, benchmarks, validation

## Develop

```bash
npm install
npm run dev        # http://localhost:3000
```

## Build

```bash
npm run build      # static export → out/
```

`out/` is a fully static site — deploy it to any static host (GitHub Pages, Vercel,
Netlify, S3, …).
