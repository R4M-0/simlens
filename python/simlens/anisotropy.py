"""Anisotropy correction (T2.2) — the corpus centroid μ and an optional whitening/ABTT map.

Real embedding spaces are anisotropic (Ethayarajh 2019): a large shared mean direction can
make *every* pair look ~97% similar, so a raw decomposition is faithful but dominated by a
baseline that carries no discriminative information. The standard fixes (Mu & Viswanath 2018;
BERT-whitening) center the space and optionally project out the top principal directions or
decorrelate the covariance. We compute μ (and, for ``abtt``/``whiten``, a linear map W) once
and store them in the bundle; the human-facing "why" then decomposes the *centered* score,
which surfaces what actually distinguishes a pair. The exact raw decomposition stays available
for auditors.

This is the same μ that Integrated Gradients uses as its baseline (T2.1) — one construct,
two uses.
"""
from __future__ import annotations

import numpy as np


def _top_pcs(Xc: np.ndarray, d_prime: int) -> np.ndarray:
    """Top-``d_prime`` principal directions of already-centered ``Xc`` (rows = samples)."""
    # SVD of the centered data; right singular vectors are the principal directions.
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return vt[:d_prime]  # [d_prime, dim]


def fit_centering(
    X: np.ndarray,
    mode: str = "abtt",
    d_prime: int | None = None,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return ``(mean μ, transform W)`` for the requested correction.

    - ``"center"``: subtract μ only (``W`` is ``None``, i.e. identity).
    - ``"abtt"``  : all-but-the-top — subtract μ, project out the top ``d_prime`` PCs
      (default ``d_prime ≈ dim/100``, per Mu & Viswanath). ``W = I − UᵀU``.
    - ``"whiten"``: subtract μ, decorrelate to identity covariance. ``W = Σ^{-1/2}``.

    A centered/whitened vector is ``(x − μ) @ W`` (or ``x − μ`` when ``W`` is ``None``).
    """
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu
    dim = X.shape[1]
    if mode == "center":
        return mu.astype(np.float32), None
    if mode == "abtt":
        dp = d_prime if d_prime is not None else max(1, round(dim / 100))
        dp = min(dp, min(Xc.shape) - 1) if min(Xc.shape) > 1 else 1
        U = _top_pcs(Xc, dp)  # [dp, dim]
        W = np.eye(dim) - U.T @ U
        return mu.astype(np.float32), W.astype(np.float32)
    if mode == "whiten":
        cov = (Xc.T @ Xc) / max(len(Xc), 1)
        vals, vecs = np.linalg.eigh(cov)
        W = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, eps))) @ vecs.T
        return mu.astype(np.float32), W.astype(np.float32)
    raise ValueError(f"unknown centering mode {mode!r}; use center|abtt|whiten")


def apply_centering(x: np.ndarray, mean: np.ndarray, whitening: np.ndarray | None) -> np.ndarray:
    """Map a raw vector into the centered/whitened space."""
    xc = np.asarray(x, dtype=np.float64).ravel() - np.asarray(mean, dtype=np.float64)
    if whitening is not None:
        xc = xc @ np.asarray(whitening, dtype=np.float64)
    return xc


def anisotropy_baseline(X: np.ndarray, metric: str = "cosine", n: int = 512, seed: int = 0) -> float:
    """Mean pairwise similarity attributable to the shared mean direction — a quick read on
    how anisotropic a space is (1.0 ≈ everything looks identical)."""
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    S = X[idx]
    if metric == "cosine":
        S = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)
    G = S @ S.T
    off = G[~np.eye(len(S), dtype=bool)]
    return float(off.mean())
