"""Dependency-free namer: label a feature by the salient tokens shared by its exemplars."""
from __future__ import annotations

import re
from collections import Counter

from .base import _STOP, _exemplar_texts

_TOKEN = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")


class KeywordNamer:
    """Names a feature from the most frequent non-stopword tokens across its exemplars."""

    source = "keyword"

    def __init__(self, max_words: int = 2):
        self.max_words = max_words

    def name(self, exemplars: list) -> tuple[str | None, float]:
        texts = _exemplar_texts(exemplars)
        if not texts:
            return None, 0.0
        counts: Counter = Counter()
        for t in texts:
            seen = {w.lower() for w in _TOKEN.findall(t)} - _STOP
            counts.update(seen)
        if not counts:
            return None, 0.0
        top = counts.most_common(self.max_words)
        # confidence = fraction of exemplars containing the top token
        conf = top[0][1] / len(texts)
        label = " ".join(w for w, _ in top)
        return label, round(float(conf), 3)
