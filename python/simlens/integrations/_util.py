"""Shared helpers for the integrations."""
from __future__ import annotations

from ..types import Attribution


def reasons_sentence(attr: Attribution, max_terms: int = 3) -> str:
    """A clean, user-facing 'why' from the concepts both items actually share.

    Unlike `Attribution.as_sentence` (a faithful full decomposition), this lists only
    positively-*shared* named concepts — the reasons a person expects — and omits the
    global-mean baseline and "both lack X" terms.
    """
    shared = [
        c
        for c in attr.contributions
        if c.polarity == "shared"
        and c.value > 0
        and c.name
        and not str(c.name).startswith(("(bias", "feat:", "dim:"))
    ]
    if not shared:
        return attr.as_sentence(max_terms)
    total = sum(abs(c.value) for c in attr.contributions) or 1.0
    parts = [f"'{c.label}' ({100 * c.value / total:.0f}%)" for c in shared[:max_terms]]
    return "Shares " + ", ".join(parts)
