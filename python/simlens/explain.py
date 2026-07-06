"""The Explainer facade — the main user-facing entry point."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import _native
from .bundle import Bundle
from .types import Attribution, Contribution


def _vec(x) -> list:
    return np.asarray(x, dtype=np.float32).ravel().tolist()


def _key(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, (list, np.ndarray)):
            h.update(np.asarray(p, dtype=np.float32).tobytes())
        else:
            h.update(str(p).encode())
    return h.hexdigest()


class Explainer:
    """Explain *why* two embeddings are similar, or why one outranks another.

    With no bundle, only exact Level-1 (dimension) attribution is available — but it
    works instantly on any vectors. Attach a Bundle to unlock feature/concept levels.
    """

    def __init__(
        self,
        bundle: Bundle | str | Path | None = None,
        metric: str = "cosine",
        cache: bool = False,
        auto_downgrade: bool = False,
    ):
        if isinstance(bundle, (str, Path)):
            bundle = Bundle.load(bundle)
        self.bundle: Bundle | None = bundle
        self.metric = bundle.metric if bundle is not None else metric
        self._sae = bundle.to_native_sae() if (bundle and bundle.has_sae) else None
        self._cav = bundle.to_native_cav() if (bundle and bundle.has_concepts) else None
        self._hash = bundle.content_hash if bundle else None
        self._auto_downgrade = auto_downgrade
        self._cache: dict | None = {} if cache else None

    # ---- level resolution -----------------------------------------------------
    def _default_level(self) -> str:
        if self._sae is not None:
            return "feature"
        if self._cav is not None:
            return "concept"
        return "dim"

    def preferred_level(self) -> str:
        """The most human-readable level available (concepts > features > dims).

        Integrations default to this — named concepts read better and avoid the raw
        feature-level anisotropy baseline.
        """
        if self._cav is not None:
            return "concept"
        if self._sae is not None:
            return "feature"
        return "dim"

    def _stamp(self, attr: Attribution) -> Attribution:
        if self._hash and attr.bundle_hash is None:
            object.__setattr__(attr, "bundle_hash", self._hash)
        return attr

    def _eff(self, x) -> list:
        """Vector as used per-metric (cosine → unit)."""
        a = np.asarray(x, dtype=np.float64).ravel()
        if self.metric == "cosine":
            n = np.linalg.norm(a)
            if n > 0:
                a = a / n
        return a.astype(np.float32).tolist()

    def _feat_index(self, con: Contribution) -> int | None:
        if con.id.startswith("feat:"):
            part = con.id.split(":", 1)[1]
            return int(part) if part.isdigit() else None
        return None

    def _exemplars_for(self, con: Contribution) -> list | None:
        if self.bundle is None:
            return None
        if con.id.startswith("feat:"):
            return self.bundle.feature_exemplars.get(con.id.split(":", 1)[1])
        if con.id.startswith("concept:") and con.name:
            return self.bundle.concept_exemplars.get(con.name)
        return None

    def _source_for(self, con: Contribution) -> str | None:
        if self.bundle is None:
            return None
        fi = self._feat_index(con)
        if fi is not None and fi < len(self.bundle.feature_source):
            return self.bundle.feature_source[fi]
        if con.id.startswith("concept:") and con.name in self.bundle.concept_names:
            k = self.bundle.concept_names.index(con.name)
            if k < len(self.bundle.concept_source):
                return self.bundle.concept_source[k]
        return None

    def _enrich(self, attr: Attribution) -> Attribution:
        """Attach exemplar evidence + name provenance to feature/concept contributions."""
        if self.bundle is None or attr.level not in ("feature", "concept"):
            return attr
        new = [
            replace(c, evidence=self._exemplars_for(c), source=self._source_for(c))
            if c.evidence is None and c.source is None
            else c
            for c in attr.contributions
        ]
        object.__setattr__(attr, "contributions", new)
        return attr

    # ---- core explain ---------------------------------------------------------
    # feature/concept attribution decomposes a *dot-product-like* score; it is only
    # meaningful for dot and cosine, not for distance metrics.
    def _require_similarity_metric(self, level: str) -> None:
        if level != "dim" and self.metric not in ("dot", "cosine"):
            raise ValueError(
                f"level={level!r} is only supported for dot/cosine metrics, not "
                f"{self.metric!r}; use level='dim' (exact for any metric)"
            )

    def explain(
        self,
        query,
        candidate,
        level: str | None = None,
        top_k: int = 8,
        min_abs: float = 0.0,
    ) -> Attribution:
        level = level or self._default_level()
        self._require_similarity_metric(level)
        q, c = _vec(query), _vec(candidate)

        ck = None
        if self._cache is not None:
            ck = _key("explain", self.metric, level, top_k, min_abs, self._hash, q, c)
            if ck in self._cache:
                return self._cache[ck]

        if level == "dim":
            d = _native.explain_l1(q, c, self.metric, top_k, min_abs)
        elif level == "feature":
            if self._sae is None:
                raise ValueError("no SAE in bundle; use level='dim' or attach a bundle")
            d = self._sae.explain(q, c, self.metric, top_k, min_abs)
        elif level == "concept":
            if self._cav is None:
                raise ValueError("no concepts in bundle; use level='dim'/'feature'")
            d = self._cav.explain(q, c, self.metric, top_k, min_abs)
        elif level == "aspect":
            return self._explain_aspect(q, c, top_k)
        else:
            raise ValueError(f"unknown level {level!r}")

        attr = self._enrich(self._stamp(Attribution.from_dict(d)))

        # Auto-downgrade to exact Level 1 when a higher level drifts too far (§13.7).
        if (
            self._auto_downgrade
            and level != "dim"
            and attr.completeness_residual > 0.05 * (abs(attr.score) or 1.0)
        ):
            attr = _native.explain_l1(q, c, self.metric, top_k, min_abs)
            attr = self._stamp(Attribution.from_dict(attr))
            object.__setattr__(
                attr,
                "warnings",
                attr.warnings + [f"auto_downgraded_to_dim (from level={level!r})"],
            )

        if ck is not None:
            self._cache[ck] = attr
        return attr

    def explain_dissimilarity(self, query, candidate, top_k: int = 8) -> Attribution:
        """Why are these *not more* similar? Surface strong one-sided SAE features (§13.4).

        The reported ``score`` is the total *one-sided feature mass* (higher = more
        dissimilar), which the contributions decompose exactly — it is deliberately not
        the raw similarity, so completeness still holds.
        """
        if self._sae is None:
            raise ValueError("dissimilarity needs an SAE bundle")
        self._require_similarity_metric("feature")
        aq = np.asarray(self._sae.encode(self._eff(query)))
        ac = np.asarray(self._sae.encode(self._eff(candidate)))
        dec = np.asarray(self._sae.dec_norm2)
        names = (self.bundle.feature_names if self.bundle else []) or []

        contribs = []
        q_only = (aq > 0) & (ac == 0)
        c_only = (ac > 0) & (aq == 0)
        for f in np.where(q_only)[0]:
            contribs.append(("query_only", int(f), float(aq[f] * dec[f])))
        for f in np.where(c_only)[0]:
            contribs.append(("candidate_only", int(f), float(ac[f] * dec[f])))
        contribs.sort(key=lambda t: t[2], reverse=True)

        total_mass = sum(v for _, _, v in contribs)  # score = total one-sided mass
        shown = [
            Contribution(
                id=f"feat:{f}",
                name=names[f] if f < len(names) else None,
                value=v,
                confidence=None,
                polarity=pol,
                evidence=self.bundle.feature_exemplars.get(str(f)) if self.bundle else None,
            )
            for pol, f, v in contribs[:top_k]
        ]
        similarity = _native.score(_vec(query), _vec(candidate), self.metric)
        shown_mass = sum(x.value for x in shown)
        return self._stamp(
            Attribution(
                score=total_mass,
                metric=self.metric,
                level="feature",
                contributions=shown,
                completeness_residual=0.0,  # φ sum exactly to `score` (the one-sided mass)
                coverage=(shown_mass / total_mass) if total_mass else 1.0,
                warnings=[
                    "dissimilarity: 'score' is total one-sided feature mass (higher = more "
                    f"dissimilar), not the raw similarity ({similarity:.3f})"
                ],
            )
        )

    def explain_batch(self, query, candidates, **kw) -> list[Attribution]:
        return [self.explain(query, c, **kw) for c in candidates]

    # ---- aspect grouping (§13.5) ---------------------------------------------
    def _explain_aspect(self, q, c, top_k: int) -> Attribution:
        if self._cav is None or not self.bundle or not self.bundle.aspects:
            raise ValueError("aspect level needs a bundle with concepts and an aspect map")
        base = Attribution.from_dict(
            self._cav.explain(q, c, self.metric, 10_000, 0.0)
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
        self._require_similarity_metric(level)
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
        contrast_score = base.score - mean_score
        residual = abs(contrast_score - sum(x.value for x in contribs))
        return self._stamp(
            Attribution(
                score=contrast_score,
                metric=self.metric,
                level=level,
                contributions=shown,
                completeness_residual=residual,
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
        self._require_similarity_metric("feature")
        return self._sae.ablate(_vec(query), _vec(candidate), self.metric, threshold)

    def steer(self, query, weights: dict[str, float]) -> np.ndarray:
        if self._cav is None or not self.bundle:
            raise ValueError("steering needs a bundle with concepts")
        idx = {name: i for i, name in enumerate(self.bundle.concept_names)}
        pairs = [(idx[k], float(w)) for k, w in weights.items() if k in idx]
        return np.asarray(self._cav.steer(_vec(query), pairs), dtype=np.float32)
