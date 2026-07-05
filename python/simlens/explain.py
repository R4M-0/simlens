"""The Explainer facade — the main user-facing entry point."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import _native
from .bundle import Bundle
from .types import Attribution, Contribution


def _vec(x) -> list:
    return np.asarray(x, dtype=np.float32).ravel().tolist()


class Explainer:
    """Explain *why* two embeddings are similar, or why one outranks another.

    With no bundle, only exact Level-1 (dimension) attribution is available — but it
    works instantly on any vectors. Attach a Bundle to unlock feature/concept levels.
    """

    def __init__(self, bundle: Bundle | str | Path | None = None, metric: str = "cosine"):
        if isinstance(bundle, (str, Path)):
            bundle = Bundle.load(bundle)
        self.bundle: Bundle | None = bundle
        self.metric = bundle.metric if bundle is not None else metric
        self._sae = bundle.to_native_sae() if (bundle and bundle.has_sae) else None
        self._cav = bundle.to_native_cav() if (bundle and bundle.has_concepts) else None
        self._hash = bundle.content_hash if bundle else None

    # ---- level resolution -----------------------------------------------------
    def _default_level(self) -> str:
        if self._sae is not None:
            return "feature"
        if self._cav is not None:
            return "concept"
        return "dim"

    def _stamp(self, attr: Attribution) -> Attribution:
        if self._hash and attr.bundle_hash is None:
            object.__setattr__(attr, "bundle_hash", self._hash)
        return attr

    # ---- core explain ---------------------------------------------------------
    def explain(
        self,
        query,
        candidate,
        level: str | None = None,
        top_k: int = 8,
        min_abs: float = 0.0,
        include_polarity: bool = True,
    ) -> Attribution:
        level = level or self._default_level()
        q, c = _vec(query), _vec(candidate)

        if level == "dim":
            d = _native.explain_l1(q, c, self.metric, top_k, min_abs, include_polarity)
        elif level == "feature":
            if self._sae is None:
                raise ValueError("no SAE in bundle; use level='dim' or attach a bundle")
            d = self._sae.explain(q, c, self.metric, top_k, min_abs, include_polarity)
        elif level == "concept":
            if self._cav is None:
                raise ValueError("no concepts in bundle; use level='dim'/'feature'")
            d = self._cav.explain(q, c, self.metric, top_k, min_abs, include_polarity)
        elif level == "aspect":
            return self._explain_aspect(q, c, top_k)
        else:
            raise ValueError(f"unknown level {level!r}")
        return self._stamp(Attribution.from_dict(d))

    def explain_batch(self, query, candidates, **kw) -> list[Attribution]:
        return [self.explain(query, c, **kw) for c in candidates]

    # ---- aspect grouping (§13.5) ---------------------------------------------
    def _explain_aspect(self, q, c, top_k: int) -> Attribution:
        if self._cav is None or not self.bundle or not self.bundle.aspects:
            raise ValueError("aspect level needs a bundle with concepts and an aspect map")
        base = Attribution.from_dict(
            self._cav.explain(q, c, self.metric, 10_000, 0.0, False)
        )
        buckets: dict[str, float] = {}
        for con in base.contributions:
            asp = self.bundle.aspects.get(con.name or con.id, "other")
            buckets[asp] = buckets.get(asp, 0.0) + con.value
        contribs = [
            Contribution(id=f"aspect:{a}", name=a, value=v, confidence=None, polarity="shared")
            for a, v in sorted(buckets.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
        ]
        total = sum(abs(x.value) for x in base.contributions) or 1.0
        cov = sum(abs(x.value) for x in contribs) / total
        return self._stamp(
            Attribution(
                score=base.score,
                metric=self.metric,
                level="aspect",
                contributions=contribs,
                completeness_residual=base.completeness_residual,
                coverage=cov,
                warnings=base.warnings,
            )
        )

    # ---- contrastive margin (§2.4) -------------------------------------------
    def explain_margin(
        self, query, better, worse, level: str | None = None, top_k: int = 8
    ) -> Attribution:
        level = level or self._default_level()
        if level == "dim":
            d = _native.explain_margin(_vec(query), _vec(better), _vec(worse), self.metric, top_k)
            return self._stamp(Attribution.from_dict(d))
        # feature/concept margin: difference of two attributions by id
        a = self.explain(query, better, level=level, top_k=10_000)
        b = self.explain(query, worse, level=level, top_k=10_000)
        return self._stamp(self._diff(a, b, level, top_k))

    def _diff(self, a: Attribution, b: Attribution, level: str, top_k: int) -> Attribution:
        acc: dict[str, tuple[str | None, float]] = {}
        for con in a.contributions:
            acc[con.id] = (con.name, acc.get(con.id, (con.name, 0.0))[1] + con.value)
        for con in b.contributions:
            name, val = acc.get(con.id, (con.name, 0.0))
            acc[con.id] = (name, val - con.value)
        contribs = [
            Contribution(id=k, name=nm, value=v, confidence=None, polarity="shared")
            for k, (nm, v) in acc.items()
            if v != 0.0
        ]
        contribs.sort(key=lambda x: abs(x.value), reverse=True)
        total = sum(abs(x.value) for x in contribs) or 1.0
        shown = contribs[:top_k]
        return Attribution(
            score=a.score - b.score,
            metric=self.metric,
            level=level,
            contributions=shown,
            completeness_residual=abs((a.score - b.score) - sum(x.value for x in contribs)),
            coverage=sum(abs(x.value) for x in shown) / total,
            warnings=[],
        )

    # ---- contrastive corpus grounding (§13.2, COCOA-inspired) -----------------
    def explain_vs_corpus(
        self, query, candidate, foil, level: str | None = None, top_k: int = 8
    ) -> Attribution:
        level = level or self._default_level()
        base = self.explain(query, candidate, level=level, top_k=10_000)
        foils = [self.explain(query, f, level=level, top_k=10_000) for f in foil]
        mean: dict[str, tuple[str | None, float]] = {}
        for fa in foils:
            for con in fa.contributions:
                nm, v = mean.get(con.id, (con.name, 0.0))
                mean[con.id] = (nm, v + con.value / max(len(foils), 1))
        contribs = []
        for con in base.contributions:
            baseline = mean.get(con.id, (con.name, 0.0))[1]
            val = con.value - baseline
            if val != 0.0:
                contribs.append(
                    Contribution(con.id, con.name, val, con.confidence, con.polarity)
                )
        contribs.sort(key=lambda x: abs(x.value), reverse=True)
        total = sum(abs(x.value) for x in contribs) or 1.0
        shown = contribs[:top_k]
        mean_score = sum(f.score for f in foils) / max(len(foils), 1)
        return self._stamp(
            Attribution(
                score=base.score - mean_score,
                metric=self.metric,
                level=level,
                contributions=shown,
                completeness_residual=0.0,
                coverage=sum(abs(x.value) for x in shown) / total,
                warnings=["contrastive: values are relative to the foil set, not the raw score"],
            )
        )

    # ---- cohort summary (§13.3) ----------------------------------------------
    def summarize(self, query, hits, level: str | None = None, top_k: int = 8) -> dict:
        level = level or self._default_level()
        agg: dict[str, dict] = {}
        for h in hits:
            a = self.explain(query, h, level=level, top_k=10_000)
            for con in a.contributions:
                e = agg.setdefault(con.id, {"name": con.name, "total": 0.0, "count": 0})
                e["total"] += con.value
                e["count"] += 1
        ranked = sorted(agg.items(), key=lambda kv: abs(kv[1]["total"]), reverse=True)
        n = len(hits)
        return {
            "n_hits": n,
            "level": level,
            "shared": [
                {
                    "id": k,
                    "name": v["name"] or k,
                    "total": v["total"],
                    "in_hits": v["count"],
                    "fraction": v["count"] / max(n, 1),
                }
                for k, v in ranked[:top_k]
            ],
        }

    # ---- ablation & steering --------------------------------------------------
    def ablate(self, query, candidate, threshold: float) -> dict:
        if self._sae is None:
            raise ValueError("ablation needs an SAE bundle")
        return self._sae.ablate(_vec(query), _vec(candidate), self.metric, threshold)

    def steer(self, query, weights: dict[str, float]) -> np.ndarray:
        if self._cav is None or not self.bundle:
            raise ValueError("steering needs a bundle with concepts")
        idx = {name: i for i, name in enumerate(self.bundle.concept_names)}
        pairs = [(idx[k], float(w)) for k, w in weights.items() if k in idx]
        return np.asarray(self._cav.steer(_vec(query), pairs), dtype=np.float32)
