"""Naming protocol shared by all namers."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

_STOP = set(
    "the a an of to and or in on at for with without from into over under is are was were "
    "be been being this that these those it its as by then than so such but not no yes will "
    "would can could may might shall should has have had do does did their his her our your".split()
)


def _exemplar_texts(exemplars: list) -> list[str]:
    """Coerce exemplar items to strings (skips non-textual ids gracefully)."""
    return [str(e) for e in (exemplars or []) if e is not None]


@runtime_checkable
class Namer(Protocol):
    source: str  # provenance tag written onto named features

    def name(self, exemplars: list) -> tuple[str | None, float]:
        """Return (label, confidence in [0,1]); (None, 0.0) if it can't name it."""
        ...
