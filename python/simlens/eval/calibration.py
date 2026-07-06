"""Confidence calibration for feature naming (§13.11).

A feature's stored confidence should mean what it says. We recompute how strongly each
named feature actually tracks its label on a (held-out) set and compare to the stored
confidence — a reliability curve — and can rewrite the confidences to the measured values.
"""
from __future__ import annotations

import numpy as np

from ..bundle import Bundle
from ..train.labeling import _corr


def _activations(bundle: Bundle, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return np.maximum(X @ np.asarray(bundle.w_enc).T + np.asarray(bundle.b_enc), 0.0)


def measured_confidence(bundle: Bundle, X: np.ndarray, labelers: dict) -> list:
    """Per-feature |correlation| with its assigned label, computed on X."""
    H = _activations(bundle, X)
    labels = {k: np.asarray(v, dtype=np.float64).ravel() for k, v in labelers.items()}
    out: list = [None] * bundle.n_features
    for f, name in enumerate(bundle.feature_names):
        if name is None or name not in labels:
            continue
        col = H[:, f]
        if col.std() == 0:
            out[f] = 0.0
        else:
            out[f] = round(abs(_corr(col, labels[name])), 4)
    return out


def reliability(bundle: Bundle, X: np.ndarray, labelers: dict, n_bins: int = 5) -> dict:
    """Bin named features by *stated* confidence and report the *measured* mean per bin.

    A well-calibrated bundle has measured ≈ stated within every bin.
    """
    measured = measured_confidence(bundle, X, labelers)
    stated = bundle.feature_conf
    rows = [
        (float(stated[f]), float(measured[f]))
        for f in range(bundle.n_features)
        if bundle.feature_names[f] is not None and measured[f] is not None and stated[f] is not None
    ]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        pts = [m for s, m in rows if (lo <= s < hi or (b == n_bins - 1 and s == hi))]
        st = [s for s, m in rows if (lo <= s < hi or (b == n_bins - 1 and s == hi))]
        bins.append(
            {
                "bin": [round(lo, 2), round(hi, 2)],
                "count": len(pts),
                "stated_mean": round(float(np.mean(st)), 4) if st else None,
                "measured_mean": round(float(np.mean(pts)), 4) if pts else None,
            }
        )
    gap = [abs(b["stated_mean"] - b["measured_mean"]) for b in bins if b["count"]]
    return {
        "n_named": len(rows),
        "bins": bins,
        "calibration_error": round(float(np.mean(gap)), 4) if gap else 0.0,
    }


def calibrate(bundle: Bundle, X: np.ndarray, labelers: dict) -> Bundle:
    """Overwrite stored feature confidences with values measured on X (in place)."""
    measured = measured_confidence(bundle, X, labelers)
    bundle.feature_conf = [
        measured[f] if measured[f] is not None else bundle.feature_conf[f]
        for f in range(bundle.n_features)
    ]
    return bundle
