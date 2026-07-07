import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "SimLens — see why your vectors match",
  description:
    "Faithful, vector-only similarity & ranking attribution — for any embedder, any vector store. The missing explanation layer for vector search.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <header className="site-header">
          <div className="inner">
            <Link href="/" className="brand">
              <img src="/logo.svg" alt="SimLens logo" />
              SimLens
            </Link>
            <nav className="nav">
              <Link href="/#features">Features</Link>
              <Link href="/how-it-works">How it works</Link>
              <Link href="/docs">Documentation</Link>
              <Link href="/docs#benchmarks">Benchmarks</Link>
              <a
                className="gh"
                href="https://github.com/ghassenov/simlens"
                target="_blank"
                rel="noreferrer"
              >
                GitHub ↗
              </a>
            </nav>
          </div>
        </header>
        {children}
        <footer className="site-footer">
          <div className="inner">
            <div>
              SimLens · Apache-2.0 · © {new Date().getFullYear()} Ghassen
              Naouar
            </div>
            <div style={{ display: "flex", gap: 18 }}>
              <Link href="/how-it-works">How it works</Link>
              <Link href="/docs">Docs</Link>
              <a
                href="https://github.com/ghassenov/simlens"
                target="_blank"
                rel="noreferrer"
              >
                GitHub
              </a>
              <a
                href="https://pypi.org/project/simlens/"
                target="_blank"
                rel="noreferrer"
              >
                PyPI
              </a>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
