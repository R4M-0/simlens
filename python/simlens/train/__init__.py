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
) -> Bundle:
    """Assemble a Bundle from a trained SAE + auto-named features."""
    X = np.asarray(corpus_vectors, dtype=np.float64)
    H = sae.encode(X)
    label_arrays = _resolve_labels(labelers, items, X.shape[0])
    names, conf = label_features(H, label_arrays, min_confidence=min_confidence)
    return Bundle(
        embedder=embedder,
        dim=sae.dim,
        metric=metric,
        modality=modality,
        w_enc=np.asarray(sae.W_enc, dtype=np.float32),
        b_enc=np.asarray(sae.b_enc, dtype=np.float32),
        dec_norm2=np.asarray(sae.dec_norm2(), dtype=np.float32),
        feature_names=names,
        feature_conf=conf,
    )


def import_sae(
    embedder: str,
    metric: str,
    w_enc: np.ndarray,
    b_enc: np.ndarray,
    w_dec: np.ndarray,
    modality: str = "text",
) -> Bundle:
    """Wrap an externally trained SAE (e.g. from SAELens/HF) into a Bundle.

    w_enc: [n_features, dim]; b_enc: [n_features]; w_dec: [dim, n_features].
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
        dec_norm2=(w_dec ** 2).sum(axis=0).astype(np.float32),
    )
