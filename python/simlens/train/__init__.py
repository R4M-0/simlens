"""Offline training: build interpretability bundles for an embedding space."""
from __future__ import annotations

import numpy as np

from ..bundle import Bundle
from .cav import fit_cav
from .labeling import label_features
from .sae import SAE

__all__ = ["SAE", "fit_cav", "label_features", "build_bundle", "import_sae"]


def _resolve_labels(labelers, items, n) -> dict:
    out = {}
    if not labelers:
        return out
    for name, fn in labelers.items():
        if callable(fn):
            if items is None:
                raise ValueError(f"labeler {name!r} is callable but no items given")
            arr = np.asarray([fn(it) for it in items], dtype=np.float64)
        else:
            arr = np.asarray(fn, dtype=np.float64)
        if arr.shape[0] != n:
            raise ValueError(f"labeler {name!r} length {arr.shape[0]} != corpus {n}")
        out[name] = arr
    return out


def build_bundle(
    embedder: str,
    metric: str,
    sae: SAE,
    corpus_vectors: np.ndarray,
    labelers: dict | None = None,
    items: list | None = None,
    modality: str = "text",
    min_confidence: float = 0.3,
    n_exemplars: int = 5,
    label_source: str = "payload",
) -> Bundle:
    """Assemble a Bundle from a trained SAE + auto-named features + exemplar evidence."""
    X = np.asarray(corpus_vectors, dtype=np.float64)
    H = sae.encode(X)
    label_arrays = _resolve_labels(labelers, items, X.shape[0])
    names, conf = label_features(H, label_arrays, min_confidence=min_confidence)
    source = [label_source if nm is not None else None for nm in names]

    # Evidence: the corpus items that most activate each *named* feature (§13.11).
    exemplars: dict = {}
    for f, name in enumerate(names):
        if name is None:
            continue
        col = H[:, f]
        top = np.argsort(col)[::-1][:n_exemplars]
        top = [int(i) for i in top if col[i] > 0]
        if top:
            exemplars[str(f)] = [items[i] if items is not None else int(i) for i in top]

    return Bundle(
        embedder=embedder,
        dim=sae.dim,
        metric=metric,
        modality=modality,
        w_enc=np.asarray(sae.W_enc, dtype=np.float32),
        b_enc=np.asarray(sae.b_enc, dtype=np.float32),
        w_dec=np.ascontiguousarray(sae.W_dec.T, dtype=np.float32),  # [n_features, dim]
        b_dec=np.asarray(sae.b_dec, dtype=np.float32),
        feature_names=names,
        feature_conf=conf,
        feature_source=source,
        feature_exemplars=exemplars,
    )


def import_sae(
    embedder: str,
    metric: str,
    w_enc: np.ndarray,
    b_enc: np.ndarray,
    w_dec: np.ndarray,
    b_dec: np.ndarray | None = None,
    modality: str = "text",
) -> Bundle:
    """Wrap an externally trained SAE (e.g. from SAELens/HF) into a Bundle.

    w_enc: [n_features, dim]; b_enc: [n_features]; w_dec: [dim, n_features];
    b_dec: [dim] (defaults to zeros).
    """
    w_enc = np.asarray(w_enc, dtype=np.float32)
    w_dec = np.asarray(w_dec, dtype=np.float32)
    dim = w_enc.shape[1]
    return Bundle(
        embedder=embedder,
        dim=dim,
        metric=metric,
        modality=modality,
        w_enc=w_enc,
        b_enc=np.asarray(b_enc, dtype=np.float32),
        w_dec=np.ascontiguousarray(w_dec.T, dtype=np.float32),  # -> [n_features, dim]
        b_dec=np.zeros(dim, np.float32) if b_dec is None else np.asarray(b_dec, np.float32),
    )
