"""Concept Activation Vectors from positive/negative example sets (TCAV-style)."""
from __future__ import annotations

import numpy as np


def fit_cav(positive: np.ndarray, negative: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (unit concept direction, linear-separability accuracy in [0,1]).

    Uses the mean-difference direction (a robust linear CAV); confidence is the accuracy
    of the induced midpoint threshold classifier on the training examples.
    """
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    direction = pos.mean(axis=0) - neg.mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm == 0:
        return direction.astype(np.float32), 0.0
    direction = direction / norm

    proj_pos = pos @ direction
    proj_neg = neg @ direction
    thresh = 0.5 * (proj_pos.mean() + proj_neg.mean())
    correct = int((proj_pos > thresh).sum()) + int((proj_neg <= thresh).sum())
    acc = correct / (len(pos) + len(neg))
    return direction.astype(np.float32), float(acc)
