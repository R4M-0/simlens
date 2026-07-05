"""Auto-name SAE features by correlating their activations with user label functions."""
from __future__ import annotations

import numpy as np


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    da, db = np.linalg.norm(a), np.linalg.norm(b)
    if da == 0 or db == 0:
        return 0.0
    return float((a @ b) / (da * db))


def label_features(
    activations: np.ndarray,
    labels: dict[str, np.ndarray],
    min_confidence: float = 0.3,
) -> tuple[list, list]:
    """Assign each feature the label it correlates with most strongly.

    activations: [N, F] SAE activations over the labeling corpus.
    labels:      name -> [N] scalar/boolean array.
    Returns (names[F], confidences[F]); unnamed features get (None, correlation).
    """
    H = np.asarray(activations, dtype=np.float64)
    f = H.shape[1]
    label_items = {k: np.asarray(v, dtype=np.float64).ravel() for k, v in labels.items()}
    names: list = [None] * f
    conf: list = [None] * f
    for j in range(f):
        col = H[:, j]
        if col.std() == 0:
            continue
        best_name, best_abs, best_signed = None, 0.0, 0.0
        for name, y in label_items.items():
            r = _corr(col, y)
            if abs(r) > best_abs:
                best_name, best_abs, best_signed = name, abs(r), r
        conf[j] = round(best_abs, 4)
        if best_abs >= min_confidence and best_signed > 0:
            names[j] = best_name
    return names, conf
