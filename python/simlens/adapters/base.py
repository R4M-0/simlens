"""Adapter protocol shared by all vector stores."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class Sample:
    """A batch drawn from a store for autofit: vectors + payloads (+ ids)."""

    vectors: np.ndarray            # [n, dim]
    payloads: list = field(default_factory=list)  # list of dicts (may be empty)
    ids: list = field(default_factory=list)


@runtime_checkable
class VectorStore(Protocol):
    """Anything that can return vectors for a list of ids, shaped [len(ids), dim]."""

    def get(self, ids: list) -> np.ndarray:  # pragma: no cover - protocol
        ...


@runtime_checkable
class SampleableStore(Protocol):
    """A store autofit can draw a training sample from."""

    def sample(self, n: int) -> Sample:  # pragma: no cover - protocol
        ...


def _require(pkg: str):
    """Import an optional dependency or raise a helpful, actionable error."""
    try:
        return __import__(pkg)
    except ImportError as e:  # pragma: no cover - env dependent
        raise ImportError(
            f"the '{pkg}' package is required for this adapter; install it with "
            f"`pip install {pkg}`"
        ) from e
