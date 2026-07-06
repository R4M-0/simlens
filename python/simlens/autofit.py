"""autofit — build a named, concept-annotated bundle by *harvesting* the host system,
so users don't have to feed a corpus, hand-label features, or curate concept examples.

    bundle = simlens.autofit(store)                       # point at your vector DB
    bundle = simlens.autofit(vectors=X, payloads=meta)    # or pass arrays

What it does, all automatically:
  1. samples vectors + payloads from the store (or takes them directly),
  2. trains an SAE on that sample,
  3. derives labelers from payload fields  → names features for free,
  4. names remaining features with a Namer  (keyword by default; LLM if provided),
  5. registers concepts from categorical payload fields and/or text phrases.
"""
from __future__ import annotations

import numpy as np

from .bundle import Bundle
from .naming.keyword import KeywordNamer
from .train import SAE, build_bundle

_TEXT_KEYS = ("text", "name", "content", "title", "smiles", "label")


def derive_payload_labelers(payloads: list[dict], max_cardinality: int = 20) -> dict:
    """Turn payload fields into labeler arrays with zero user effort.

    - numeric field           → the value per item (e.g. logp, mw)
    - boolean field           → 0/1 per item        (e.g. approved)
    - low-cardinality string  → one 0/1 labeler per value (e.g. topic=finance)
    - list-of-tags field      → one 0/1 labeler per tag
    High-cardinality / free-text fields are skipped.
    """
    n = len(payloads)
    if n == 0:
        return {}
    keys = set()
    for p in payloads:
        keys.update(p.keys())

    out: dict[str, np.ndarray] = {}
    for k in keys:
        vals = [p.get(k) for p in payloads]
        present = [v for v in vals if v is not None]
        if not present:
            continue
        sample = present[0]
        if isinstance(sample, bool):
            out[k] = np.array([1.0 if v else 0.0 for v in vals])
        elif isinstance(sample, (int, float)):
            out[k] = np.array([float(v) if isinstance(v, (int, float)) else np.nan for v in vals])
        elif isinstance(sample, str):
            uniq = {v for v in present if isinstance(v, str)}
            if 1 < len(uniq) <= max_cardinality:
                for val in uniq:
                    out[f"{k}={val}"] = np.array([1.0 if v == val else 0.0 for v in vals])
        elif isinstance(sample, (list, tuple)):
            tags = {t for v in present if isinstance(v, (list, tuple)) for t in v}
            if len(tags) <= max_cardinality:
                for t in tags:
                    out[f"{k}:{t}"] = np.array(
                        [1.0 if isinstance(v, (list, tuple)) and t in v else 0.0 for v in vals]
                    )
    # drop labelers that are constant or all-NaN (uninformative)
    return {k: v for k, v in out.items() if np.nanstd(v) > 0}


def _items_from_payloads(payloads: list[dict]) -> list | None:
    for key in _TEXT_KEYS:
        if any(key in p for p in payloads):
            return [str(p.get(key, "")) for p in payloads]
    return None


def autofit(
    store=None,
    *,
    vectors: np.ndarray | None = None,
    payloads: list[dict] | None = None,
    items: list | None = None,
    embedder: str = "unknown",
    metric: str = "cosine",
    sample: int = 20_000,
    expansion: int = 8,
    epochs: int = 60,
    l1: float = 1e-3,
    sae: SAE | None = None,
    labelers: dict | None = None,
    namer=None,
    text_concepts: dict | None = None,
    embed_text=None,
    discover_concepts: bool = True,
    min_confidence: float = 0.3,
    seed: int = 0,
) -> Bundle:
    """Build a fully-named bundle with minimal user input. See module docstring."""
    # 1. resolve data --------------------------------------------------------
    if store is not None:
        s = store.sample(sample)
        vectors, payloads = s.vectors, s.payloads
    if vectors is None:
        raise ValueError("provide a `store` or `vectors=`")
    X = np.asarray(vectors, dtype=np.float32)
    payloads = payloads or [{} for _ in range(len(X))]
    items = items or _items_from_payloads(payloads)

    # 2. train (or reuse) the SAE -------------------------------------------
    if sae is None:
        sae = SAE(dim=X.shape[1], expansion=expansion, l1=l1, seed=seed).fit(X, epochs=epochs)

    # 3. payload labelers (+ any user ones) → free feature names ------------
    auto = derive_payload_labelers(payloads)
    all_labelers = {**auto, **(labelers or {})}
    bundle = build_bundle(
        embedder, metric, sae, X, labelers=all_labelers, items=items, min_confidence=min_confidence
    )

    # 4. name the still-unnamed features with a Namer -----------------------
    namer = namer if namer is not None else KeywordNamer()
    for f, name in enumerate(bundle.feature_names):
        if name is not None:
            continue
        exemplars = bundle.feature_exemplars.get(str(f)) or _top_exemplars(sae, X, items, f)
        if not exemplars:
            continue
        label, conf = namer.name(exemplars)
        if label:
            bundle.feature_names[f] = label
            bundle.feature_conf[f] = conf
            if f < len(bundle.feature_source):
                bundle.feature_source[f] = getattr(namer, "source", "ai")
            bundle.feature_exemplars.setdefault(str(f), list(exemplars)[:5])

    # 5. concepts -----------------------------------------------------------
    if discover_concepts:
        _concepts_from_payload(bundle, X, auto)
    if text_concepts:
        if embed_text is None:
            raise ValueError("text_concepts require an `embed_text` callable")
        for name, phrase in text_concepts.items():
            phrases = phrase if isinstance(phrase, (list, tuple)) else [phrase]
            direction = np.mean([np.asarray(embed_text(p), np.float32).ravel() for p in phrases], axis=0)
            bundle.add_text_concept(name, direction, aspect="text")

    # stamp provenance so explanations carry a hash even before the bundle is saved
    bundle.content_hash = bundle.compute_hash()
    return bundle


def _top_exemplars(sae: SAE, X: np.ndarray, items, f: int, k: int = 5) -> list:
    if items is None:
        return []
    col = sae.encode(X)[:, f]
    top = np.argsort(col)[::-1][:k]
    return [items[i] for i in top if col[i] > 0]


def _concepts_from_payload(bundle: Bundle, X: np.ndarray, labelers: dict, min_examples: int = 8):
    """Register a concept from each binary payload labeler with enough +/- examples."""
    for name, arr in labelers.items():
        vals = np.asarray(arr)
        if not set(np.unique(vals[~np.isnan(vals)])).issubset({0.0, 1.0}):
            continue  # only clean binary fields become concepts
        pos = X[vals == 1.0]
        neg = X[vals == 0.0]
        if len(pos) >= min_examples and len(neg) >= min_examples and name not in bundle.concept_names:
            bundle.add_concept(name, pos, neg, aspect="payload")
