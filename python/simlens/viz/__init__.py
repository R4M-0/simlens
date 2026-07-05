"""Terminal-friendly rendering of an Attribution (no external deps)."""
from __future__ import annotations

from ..types import Attribution


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


__all__ = ["render"]
