"""Offline training: build interpretability bundles for an embedding space."""
from __future__ import annotations

import numpy as np

from ..bundle import Bundle
from .cav import fit_cav
from .labeling import label_features
from .metrics import sae_quality
from .sae import (
    _ARCHES,
    SAE,
    BatchTopKSAE,
    JumpReLUSAE,
    ReLUSAE,
    TopKSAE,
    geometric_median,
)

__all__ = [
    "SAE",
    "ReLUSAE",
    "TopKSAE",
    "BatchTopKSAE",
    "JumpReLUSAE",
    "fit",
    "fit_cav",
    "label_features",
    "build_bundle",
    "import_sae",
    "import_safetensors_sae",
    "sae_quality",
    "geometric_median",
]


def fit(
    vectors: np.ndarray,
    arch: str = "topk",
    backend: str = "auto",
    k: int = 32,
    expansion: int = 8,
    epochs: int = 60,
    seed: int = 0,
    verbose: bool = False,
    **kw,
):
    """Train an SAE on ``vectors`` and return it. The recommended entry point.

    ``arch``: ``"topk"`` (default), ``"batchtopk"``, ``"jumprelu"`` or ``"relu"``.
    ``backend``: ``"numpy"`` (default, always available) or ``"torch"`` (needs
    ``pip install simlens[train]``); ``"auto"`` prefers torch when installed.
    ``k`` sets the target L0 for the top-k architectures.
    """
    X = np.asarray(vectors, dtype=np.float64)
    if arch not in _ARCHES:
        raise ValueError(f"unknown arch {arch!r}; choose from {sorted(_ARCHES)}")

    if backend == "auto":
        try:
            import torch  # noqa: F401

            backend = "torch"
        except ImportError:
            backend = "numpy"
    if backend == "torch":
        from .torch_backend import fit_torch

        return fit_torch(X, arch=arch, k=k, expansion=expansion, epochs=epochs, seed=seed,
                         verbose=verbose, **kw)

    cls = _ARCHES[arch]
    if arch in ("topk", "batchtopk"):
        kw.setdefault("k", k)
    sae = cls(dim=X.shape[1], expansion=expansion, seed=seed, **kw)
    return sae.fit(X, epochs=epochs, verbose=verbose)


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
    sae,
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

    threshold = getattr(sae, "threshold", None)
    threshold = None if threshold is None else np.asarray(threshold, dtype=np.float32)
    return Bundle(
        embedder=embedder,
        dim=sae.dim,
        metric=metric,
        modality=modality,
        w_enc=np.asarray(sae.W_enc, dtype=np.float32),
        b_enc=np.asarray(sae.b_enc, dtype=np.float32),
        w_dec=np.ascontiguousarray(np.asarray(sae.W_dec).T, dtype=np.float32),  # [n_features, dim]
        b_dec=np.asarray(sae.b_dec, dtype=np.float32),
        sae_k=int(getattr(sae, "k", 0) or 0),
        sae_threshold=threshold,
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
    threshold: np.ndarray | None = None,
    k: int = 0,
    modality: str = "text",
) -> Bundle:
    """Wrap an externally trained SAE (e.g. from SAELens/HF) into a Bundle.

    w_enc: [n_features, dim]; b_enc: [n_features]; w_dec: [dim, n_features];
    b_dec: [dim] (defaults to zeros). ``threshold`` (JumpReLU per-feature gate) and ``k``
    (top-k gate) are optional and reproduce the source model's sparsity at inference.
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
        sae_k=int(k),
        sae_threshold=None if threshold is None else np.asarray(threshold, np.float32),
    )


# Common SAELens / HF safetensors key aliases → our names.
_ST_KEYS = {
    "w_enc": ["W_enc", "w_enc", "encoder.weight"],
    "b_enc": ["b_enc", "b_enc.weight", "encoder.bias"],
    "w_dec": ["W_dec", "w_dec", "decoder.weight"],
    "b_dec": ["b_dec", "b_dec.weight", "decoder.bias"],
    "threshold": ["threshold", "jumprelu.threshold", "log_threshold"],
}


def import_safetensors_sae(
    embedder: str,
    metric: str,
    path: str,
    k: int = 0,
    modality: str = "text",
) -> Bundle:
    """Load a SAELens/HF ``.safetensors`` SAE and wrap it into a Bundle.

    Needs ``pip install simlens[train]`` (safetensors). Handles the common key aliases and
    the two decoder orientations ([dim, n_features] or [n_features, dim]).
    """
    try:
        from safetensors.numpy import load_file
    except ImportError as e:  # pragma: no cover
        raise ImportError("safetensors is required; `pip install simlens[train]`") from e

    tensors = load_file(path)

    def pick(kind):
        for key in _ST_KEYS[kind]:
            if key in tensors:
                return np.asarray(tensors[key])
        return None

    w_enc = pick("w_enc")
    w_dec = pick("w_dec")
    if w_enc is None or w_dec is None:
        raise ValueError(f"could not find encoder/decoder in {list(tensors)}")
    # normalize orientation: want w_enc [n_features, dim], w_dec [dim, n_features]
    b_enc = pick("b_enc")
    n_features = b_enc.shape[0] if b_enc is not None else max(w_enc.shape)
    if w_enc.shape[0] != n_features:
        w_enc = w_enc.T
    dim = w_enc.shape[1]
    if w_dec.shape[0] != dim:
        w_dec = w_dec.T
    b_dec = pick("b_dec")
    thr = pick("threshold")
    # some JumpReLU checkpoints store the log-threshold instead
    if "log_threshold" in tensors:
        thr = np.exp(np.asarray(tensors["log_threshold"]))
    return import_sae(
        embedder, metric, w_enc,
        np.zeros(n_features, np.float32) if b_enc is None else b_enc,
        w_dec, b_dec, threshold=thr, k=k, modality=modality,
    )
