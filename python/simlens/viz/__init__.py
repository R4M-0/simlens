"""Terminal-friendly rendering of an Attribution + token heatmaps (no external deps)."""
from __future__ import annotations

from ..types import Attribution

_BLOCKS = " ▁▂▃▄▅▆▇█"


def render(attr: Attribution, width: int = 28) -> str:
    """ASCII bar chart of contributions, signed and scaled to the largest magnitude."""
    lines = [
        f"{attr.level} attribution  score={attr.score:.3f}  "
        f"residual={attr.completeness_residual:.2e}  coverage={attr.coverage:.0%}"
    ]
    if not attr.contributions:
        lines.append("  (no contributions above threshold)")
    else:
        peak = max(abs(c.value) for c in attr.contributions) or 1.0
        for c in attr.contributions:
            n = int(round(width * abs(c.value) / peak))
            bar = ("█" if c.value >= 0 else "░") * max(n, 1)
            conf = f" ~{c.confidence:.2f}" if c.confidence is not None else ""
            lines.append(f"  {c.label[:24]:<24} {c.value:+.4f}{conf}  {bar}")
    for w in attr.warnings:
        lines.append(f"  ⚠ {w}")
    return "\n".join(lines)


def highlight(tokens: list, scores: list, mode: str = "blocks") -> str:
    """Render a per-token importance heatmap.

    mode="blocks": inline unicode intensity under each token (terminal-safe).
    mode="html":   <span> with background opacity keyed to score (for notebooks).
    """
    scores = list(scores)
    peak = max((abs(s) for s in scores), default=0.0) or 1.0
    if mode == "html":
        spans = []
        for tok, s in zip(tokens, scores):
            a = min(1.0, abs(s) / peak)
            color = f"rgba(79,70,229,{a:.2f})"  # indigo, opacity = importance
            spans.append(f'<span style="background:{color};padding:1px 3px">{tok}</span>')
        return " ".join(spans)
    # blocks
    cells = []
    for tok, s in zip(tokens, scores):
        idx = int(round((abs(s) / peak) * (len(_BLOCKS) - 1)))
        cells.append(f"{tok}{_BLOCKS[idx]}")
    return " ".join(cells)


__all__ = ["render", "highlight"]
