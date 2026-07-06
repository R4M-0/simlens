"""Shared data helpers for the validation notebooks (T1.2).

Each notebook tries to use a *real* embedder / dataset and falls back to an offline,
deterministic substitute so the scripts run anywhere (CI, a laptop with no network). The
fallback is a hashing bag-of-words embedder — crude, but it produces genuinely topical,
interpretable embeddings, so the explanations are real, not mocked.
"""
from __future__ import annotations

import hashlib

import numpy as np

_STOP = set("the a an of to and or in on at for with is are was be this that it as by".split())


def hashing_embed(texts: list[str], dim: int = 256) -> np.ndarray:
    """Deterministic hashing bag-of-words embedder (offline fallback), L2-normalized."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in str(t).lower().split():
            w = "".join(ch for ch in w if ch.isalnum())
            if not w or w in _STOP:
                continue
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            out[i, h % dim] += 1.0
            out[i, (h // dim) % dim] += 0.5  # a second bucket reduces collisions
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def embed(texts: list[str], dim: int = 256) -> tuple[np.ndarray, str]:
    """Embed with fastembed (bge-small) if installed, else the hashing fallback.

    Returns (vectors, backend_name).
    """
    try:
        from fastembed import TextEmbedding

        model = TextEmbedding("BAAI/bge-small-en-v1.5")
        vecs = np.asarray(list(model.embed(texts)), dtype=np.float32)
        vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
        return vecs, "fastembed/bge-small-en-v1.5"
    except Exception:  # noqa: BLE001 — offline fallback
        return hashing_embed(texts, dim=dim), f"hashing-bow(dim={dim})"


# A small, topical document set used by every notebook when no external dataset is present.
_TOPICS = {
    "finance": [
        "quarterly earnings beat expectations as revenue grew",
        "the central bank raised interest rates to curb inflation",
        "investors sold stocks amid recession fears in the market",
        "the company reported strong profit and dividend growth",
        "bond yields rose sharply after the treasury auction",
    ],
    "sports": [
        "the striker scored a hat trick in the championship final",
        "the team won the league title after a dramatic season",
        "the coach praised the defense after a clean sheet victory",
        "the tennis player advanced to the grand slam semifinal",
        "the marathon runner set a new world record time",
    ],
    "medicine": [
        "the trial showed the drug reduced tumor growth in patients",
        "researchers discovered a gene linked to the disease",
        "the vaccine produced a strong immune response in the study",
        "doctors recommend the therapy for chronic heart conditions",
        "the clinic reported fewer infections after the treatment",
    ],
    "technology": [
        "the startup launched a new machine learning platform",
        "engineers released an open source database with faster queries",
        "the chip delivers higher performance at lower power",
        "the app uses neural networks to recommend content",
        "developers adopted the framework for building web services",
    ],
}


def topic_corpus(repeat: int = 6, dim: int = 256):
    """A labeled multi-topic corpus: (vectors, texts, payloads, labels, topics, backend)."""
    texts, labels, topics = [], [], list(_TOPICS)
    for r in range(repeat):
        for ti, (topic, sents) in enumerate(_TOPICS.items()):
            s = sents[r % len(sents)]
            texts.append(f"{s} ({topic} #{r})")
            labels.append(ti)
    vecs, backend = embed(texts, dim=dim)
    payloads = [{"text": texts[i], "topic": topics[labels[i]]} for i in range(len(texts))]
    return vecs, texts, payloads, np.asarray(labels), topics, backend
