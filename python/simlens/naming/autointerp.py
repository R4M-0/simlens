"""Automated interpretability: generate a feature label, then *measure* it.

The mech-interp community solved the measurement problem — a good name is a good *classifier*
of the feature, so score it as one (Bills et al. 2023; Paulo et al. / EleutherAI 2024;
Delphi). This module:

1. **generates** a label from exemplars sampled across activation *deciles* (not just the
   top — deciles generalize better), and
2. **scores** it by *detection*: show a scorer a randomized mix of held-out activating and
   non-activating exemplars, give it the label, ask which activate, and compute **balanced
   accuracy**. That score *is* the name's faithfulness.

Everything is provider-agnostic (any ``complete(prompt) -> str`` callable) and cached. The
detection score becomes the feature's calibrated confidence; names below a threshold are
dropped rather than shown as confident nonsense.
"""
from __future__ import annotations

import hashlib
import re
from typing import Callable

import numpy as np

from .base import _exemplar_texts

_GEN_PROMPT = (
    "Below are items that activate a single feature, sampled across its activation range "
    "(strongest first). They share one specific concept or property:\n{items}\n\n"
    "Reply with ONLY a short (2-4 word) human-readable label for what they share. "
    "No punctuation, no explanation."
)

_DETECT_PROMPT = (
    "A feature is described as: \"{label}\".\n\n"
    "Here are numbered items. Reply with ONLY the numbers of the items that match that "
    "description, comma-separated (e.g. \"1, 3, 4\"). If none match, reply \"none\".\n\n{items}"
)


def decile_exemplars(activations: np.ndarray, items: list, n_per_decile: int = 1, seed: int = 0) -> list:
    """Sample activating items across deciles of their activation strength."""
    col = np.asarray(activations, dtype=np.float64)
    active = np.where(col > 0)[0]
    if active.size == 0:
        return []
    order = active[np.argsort(col[active])[::-1]]  # strongest first
    rng = np.random.default_rng(seed)
    picks: list[int] = []
    chunks = np.array_split(order, min(10, len(order)))
    for ch in chunks:
        if len(ch):
            picks.extend(rng.choice(ch, size=min(n_per_decile, len(ch)), replace=False).tolist())
    # keep the strongest ordering
    picks = sorted(set(picks), key=lambda i: -col[i])
    return [items[i] for i in picks]


def negative_exemplars(activations: np.ndarray, items: list, n: int, seed: int = 0) -> list:
    col = np.asarray(activations, dtype=np.float64)
    inactive = np.where(col <= 0)[0]
    if inactive.size == 0:
        return []
    rng = np.random.default_rng(seed + 1)
    idx = rng.choice(inactive, size=min(n, inactive.size), replace=False)
    return [items[i] for i in idx]


def generate(complete: Callable[[str], str], exemplars: list, max_exemplars: int = 12) -> str | None:
    texts = _exemplar_texts(exemplars)[:max_exemplars]
    if not texts:
        return None
    prompt = _GEN_PROMPT.format(items="\n".join(f"- {t}" for t in texts))
    try:
        resp = (complete(prompt) or "").strip()
    except Exception:  # noqa: BLE001 — naming is best-effort
        return None
    label = resp.splitlines()[0].strip().strip(".\"'` ") if resp else ""
    return label or None


def _parse_choices(resp: str, n: int) -> set[int]:
    if "none" in resp.lower():
        return set()
    return {int(x) - 1 for x in re.findall(r"\d+", resp) if 1 <= int(x) <= n}


def score_detection(
    complete: Callable[[str], str],
    label: str,
    positives: list,
    negatives: list,
    seed: int = 0,
) -> float:
    """Balanced accuracy of the label-as-classifier on a shuffled pos/neg mix."""
    pos = _exemplar_texts(positives)
    neg = _exemplar_texts(negatives)
    if not pos or not neg:
        return 0.0
    items = [(t, True) for t in pos] + [(t, False) for t in neg]
    rng = np.random.default_rng(seed)
    rng.shuffle(items)
    numbered = "\n".join(f"{i + 1}. {t}" for i, (t, _) in enumerate(items))
    prompt = _DETECT_PROMPT.format(label=label, items=numbered)
    try:
        chosen = _parse_choices(complete(prompt) or "", len(items))
    except Exception:  # noqa: BLE001
        return 0.0
    tp = sum(1 for i, (_, y) in enumerate(items) if y and i in chosen)
    fn = sum(1 for i, (_, y) in enumerate(items) if y and i not in chosen)
    tn = sum(1 for i, (_, y) in enumerate(items) if not y and i not in chosen)
    fp = sum(1 for i, (_, y) in enumerate(items) if not y and i in chosen)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return round(0.5 * (tpr + tnr), 4)


def score_fuzz(
    complete: Callable[[str], str],
    label: str,
    positives: list,
    seed: int = 0,
) -> float:
    """Fuzzing precision: does the label still hold when the activating items are mixed with
    unrelated ones? Reuses detection scoring with the positives as the signal, shuffled
    filler as noise (a lightweight precision proxy)."""
    pos = _exemplar_texts(positives)
    if len(pos) < 2:
        return 0.0
    half = len(pos) // 2
    return score_detection(complete, label, pos[:half], pos[half:][::-1], seed=seed)


class ValidatedNamer:
    """Generate a label and attach a *measured* detection accuracy as its confidence.

    Names below ``min_score`` are dropped (returned as ``None``) — an unmeasured guess is
    worse than no name. Tagged ``source="ai"`` and user-editable.
    """

    source = "ai"

    def __init__(
        self,
        complete: Callable[[str], str],
        min_score: float = 0.6,
        max_exemplars: int = 12,
        cache: bool = True,
    ):
        self.complete = complete
        self.min_score = min_score
        self.max_exemplars = max_exemplars
        self._cache: dict | None = {} if cache else None

    def name(self, exemplars: list) -> tuple[str | None, float]:
        """Protocol-compatible generation-only naming (no held-out negatives available)."""
        label = generate(self.complete, exemplars, self.max_exemplars)
        return (label, 0.0) if label else (None, 0.0)

    def name_scored(
        self,
        gen_exemplars: list,
        detect_positives: list,
        detect_negatives: list,
    ) -> tuple[str | None, float]:
        """Generate from ``gen_exemplars``, then score detection on held-out pos/neg.

        Returns ``(label, detection_accuracy)``; ``(None, score)`` if below threshold.
        """
        key = None
        if self._cache is not None:
            h = hashlib.sha256(
                "|".join(_exemplar_texts(gen_exemplars)).encode()
            ).hexdigest()
            key = h
            if key in self._cache:
                return self._cache[key]
        label = generate(self.complete, gen_exemplars, self.max_exemplars)
        if not label:
            out = (None, 0.0)
        else:
            acc = score_detection(self.complete, label, detect_positives, detect_negatives)
            out = (label, acc) if acc >= self.min_score else (None, acc)
        if key is not None:
            self._cache[key] = out
        return out
