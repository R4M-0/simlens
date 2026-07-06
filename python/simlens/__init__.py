"""SimLens — faithful, vector-only similarity & ranking attribution.

The 'why' that vector search never returns: given two embeddings and a metric, decompose
the similarity score into additive, completeness-checked contributions at three zoom
levels — dimensions (exact), SAE features, and named concepts.

Quick start (zero setup, exact Level-1)::

    import simlens
    ex = simlens.Explainer(metric="cosine")
    attr = ex.explain(query_vec, candidate_vec)
    print(attr.as_sentence())
"""
from __future__ import annotations

from . import adapters, anisotropy, eval, integrations, naming, train, viz
from ._native import score
from .anisotropy import apply_centering, fit_centering
from .autofit import autofit
from .bundle import Bundle
from .explain import Explainer
from .hierarchy import discover_hierarchy
from .learned import LearnedMetricExplainer
from .multivector import MultiVectorExplainer
from .types import Attribution, Contribution

__version__ = "0.2.0"

__all__ = [
    "Explainer",
    "MultiVectorExplainer",
    "LearnedMetricExplainer",
    "Bundle",
    "Attribution",
    "Contribution",
    "autofit",
    "score",
    "fit_centering",
    "apply_centering",
    "discover_hierarchy",
    "train",
    "eval",
    "adapters",
    "anisotropy",
    "integrations",
    "naming",
    "viz",
    "__version__",
]
