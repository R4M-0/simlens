"""Integrated Gradients for learned / non-linear similarity metrics (T2.1).

SimLens decomposes dot / cosine / euclidean *exactly*. Production rerankers — cross-encoders,
bi-encoders with a learned head, arbitrary ``score(q, c)`` callbacks — are opaque to that.
Integrated Gradients (Sundararajan et al., 2017) extends attribution to any differentiable
scorer with a completeness axiom::

    φ_i = (q_i − q0_i) · ∫₀¹ ∂ s(q0 + t·(q−q0), c)/∂q_i dt        (Riemann sum over m steps)
    Σ_i φ_i = s(q, c) − s(q0, c)                                  (completeness)

The baseline ``q0`` is the crux and is well-studied: for *similarity* the principled choice
is the **corpus centroid μ** (a.k.a. "average background"), so IG attributes the *excess
similarity over a typical item*, ``Σφ = s(q,c) − s(μ,c)``. This is the same μ as the
anisotropy correction (T2.2) — one construct, two uses.

Gradients come from torch autodiff when available, or a finite-difference fallback that works
for *any* black-box callable. Path/SmoothGrad averaging mitigates shattered gradients.
"""
from __future__ import annotations

import numpy as np

from .types import Attribution, Contribution


class LearnedMetricExplainer:
    """Explain any differentiable (or black-box) ``score(q, c)`` via Integrated Gradients.

    ``scorer(q, c) -> float`` takes two vectors (numpy, or torch tensors if ``grad="torch"``).
    ``grad``: ``None`` → finite differences (robust, works on any callable); ``"torch"`` →
    autodiff (scorer must accept torch tensors); or a callable ``grad(x, c) -> ∂s/∂x``.
    ``baseline``: ``"centroid"`` (needs ``mean=``), ``"zero"``, or an explicit vector.
    """

    def __init__(
        self,
        scorer,
        grad=None,
        baseline="centroid",
        mean: np.ndarray | None = None,
        steps: int = 32,
        eps: float = 1e-3,
        metric: str = "learned",
    ):
        self.scorer = scorer
        self.grad = grad
        self.baseline = baseline
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float64).ravel()
        self.steps = int(steps)
        self.eps = float(eps)
        self.metric = metric

    # ---- baseline & gradients -------------------------------------------------
    def _q0(self, q: np.ndarray) -> np.ndarray:
        b = self.baseline
        if isinstance(b, str):
            if b == "zero":
                return np.zeros_like(q)
            if b == "centroid":
                if self.mean is None:
                    raise ValueError(
                        "baseline='centroid' needs the corpus mean; pass mean=bundle.mean "
                        "or LearnedMetricExplainer(..., mean=μ)"
                    )
                return self.mean.astype(q.dtype)
            raise ValueError(f"unknown baseline {b!r}")
        return np.asarray(b, dtype=q.dtype).ravel()

    def _score(self, q, c) -> float:
        return float(self.scorer(q, c))

    def _gradient(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        if callable(self.grad):
            return np.asarray(self.grad(x, c), dtype=np.float64).ravel()
        if self.grad == "torch":
            return self._torch_gradient(x, c)
        return self._finite_diff(x, c)

    def _finite_diff(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Central finite differences — a gradient for any black-box scalar scorer."""
        g = np.zeros_like(x)
        h = self.eps
        for i in range(x.size):
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            g[i] = (self._score(xp, c) - self._score(xm, c)) / (2.0 * h)
        return g

    def _torch_gradient(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        import torch

        xt = torch.tensor(np.asarray(x, dtype=np.float32), requires_grad=True)
        ct = torch.tensor(np.asarray(c, dtype=np.float32))
        s = self.scorer(xt, ct)
        s.backward()
        return xt.grad.detach().cpu().numpy().astype(np.float64).ravel()

    # ---- attribution ----------------------------------------------------------
    def explain(
        self,
        query,
        candidate,
        top_k: int = 8,
        min_abs: float = 0.0,
        n_paths: int = 1,
        noise: float = 0.0,
        seed: int = 0,
    ) -> Attribution:
        q = np.asarray(query, dtype=np.float64).ravel()
        c = np.asarray(candidate, dtype=np.float64).ravel()
        q0 = self._q0(q)
        rng = np.random.default_rng(seed)

        total = np.zeros_like(q)
        for _ in range(max(1, n_paths)):
            base = q0 + noise * rng.standard_normal(q0.shape) if noise else q0
            acc = np.zeros_like(q)
            for step in range(self.steps):
                t = (step + 0.5) / self.steps  # midpoint Riemann rule
                acc += self._gradient(base + t * (q - base), c)
            total += (acc / self.steps) * (q - base)
        phi = total / max(1, n_paths)

        raw = self._score(q, c)
        base_score = self._score(q0, c)
        target = raw - base_score  # what completeness says Σφ should equal
        residual = abs(target - float(phi.sum()))

        order = np.argsort(np.abs(phi))[::-1]
        total_abs = float(np.abs(phi).sum()) or 1.0
        shown = [i for i in order if abs(phi[i]) >= min_abs][:top_k]
        contribs = [
            Contribution(
                id=f"dim:{int(i)}",
                name=None,
                value=float(phi[i]),
                confidence=None,
                polarity="shared" if phi[i] >= 0 else "neither",
            )
            for i in shown
        ]
        coverage = sum(abs(x.value) for x in contribs) / total_abs
        warnings = [
            f"integrated_gradients: Σφ = s(q,c) − s(baseline,c) = {target:.4f} "
            f"(raw score {raw:.4f}); baseline={self.baseline!r}",
        ]
        if residual > 0.05 * (abs(target) or 1.0):
            warnings.append(
                f"ig_completeness_residual: {residual:.4f}; increase steps for a tighter bound"
            )
        return Attribution(
            score=target,
            metric=self.metric,
            level="dim",
            contributions=contribs,
            completeness_residual=residual,
            coverage=coverage,
            warnings=warnings,
        )

    def deletion_curve(self, query, candidate, k: int = 20, seed: int = 0) -> dict:
        """Delete top-attributed dims in order; a faithful ranking drops the score faster
        than random (RISE-style). Lower top-AUC than random ⇒ faithful."""
        q = np.asarray(query, dtype=np.float64).ravel()
        c = np.asarray(candidate, dtype=np.float64).ravel()
        a = self.explain(q, c, top_k=q.size)
        order = [int(con.id.split(":")[1]) for con in a.contributions]

        def curve(idx_order):
            qq = q.copy()
            scores = [self._score(qq, c)]
            for i in idx_order[:k]:
                qq[i] = 0.0
                scores.append(self._score(qq, c))
            return scores

        rng = np.random.default_rng(seed)
        rand = list(range(q.size))
        rng.shuffle(rand)
        top = curve(order)
        base = curve(rand)
        return {
            "auc_top": float(np.trapezoid(top)),
            "auc_random": float(np.trapezoid(base)),
            "faithful": float(np.trapezoid(top)) < float(np.trapezoid(base)),
        }
