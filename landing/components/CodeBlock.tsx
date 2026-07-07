import React from "react";

/**
 * Minimal syntax-styled code block. `children` is pre-tokenized JSX so we
 * don't need a highlighter dependency; use the helper spans below.
 */
export function CodeBlock({
  title,
  children,
}: {
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="codeblock">
      <div className="cb-head">
        <span className="dot" style={{ background: "#ef4444" }} />
        <span className="dot" style={{ background: "#eab308" }} />
        <span className="dot" style={{ background: "#22c55e" }} />
        {title && <span className="cb-title">{title}</span>}
      </div>
      <pre>{children}</pre>
    </div>
  );
}

export const Kw = ({ children }: { children: React.ReactNode }) => (
  <span className="c-kw">{children}</span>
);
export const Str = ({ children }: { children: React.ReactNode }) => (
  <span className="c-str">{children}</span>
);
export const Cm = ({ children }: { children: React.ReactNode }) => (
  <span className="c-comment">{children}</span>
);
export const Fn = ({ children }: { children: React.ReactNode }) => (
  <span className="c-fn">{children}</span>
);
export const Out = ({ children }: { children: React.ReactNode }) => (
  <span className="c-out">{children}</span>
);
