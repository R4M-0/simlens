"""Sparse-autoencoder trainers (numpy, dependency-light).

Three architectures that the interpretability field converged on, plus the original plain
ReLU one:

- ``TopKSAE`` (default) — keep only the top-``k`` activations per row; ``k`` *directly* sets
  L0, so there is no fragile L1 coefficient to tune. An **AuxK** auxiliary loss reconstructs
  the residual from dead latents to revive them (Gao et al., 2024).
- ``BatchTopKSAE`` — top-k taken across the whole batch (Bussmann et al., 2024); at
  inference it falls back to a per-row threshold so it stays a pure feed-forward gate.
- ``JumpReLUSAE`` — a learnable per-feature Heaviside gate trained with straight-through
  estimators (Rajamanoharan et al., 2024).
- ``ReLUSAE`` (aliased ``SAE``) — the original plain ReLU + L1 autoencoder, kept as a
  zero-knob fallback and for backward compatibility.

All expose the same surface (``W_enc`` ``[f,d]``, ``b_enc`` ``[f]``, ``W_dec`` ``[d,f]``,
``b_dec`` ``[d]``, ``encode`` / ``decode``) and, crucially, an ``encode`` that reproduces the
*training-time* sparsity gate — so the activations used for attribution are exactly the ones
the trainer optimized (train == inference). The gate (``k`` / ``threshold``) is carried into
the bundle and the Rust kernel.

For research-scale dictionaries use the optional torch backend (``pip install
simlens[train]``, ``backend="torch"``); the numpy path stays the tested default.
"""
from __future__ import annotations

import numpy as np

from .metrics import sae_quality


def _relu(x):
    return np.maximum(x, 0.0)


def geometric_median(X: np.ndarray, iters: int = 64, eps: float = 1e-8) -> np.ndarray:
    """Weiszfeld geometric median — a robust decoder-bias init (Gao et al. recipe)."""
    X = np.asarray(X, dtype=np.float64)
    y = X.mean(axis=0)
    for _ in range(iters):
        d = np.linalg.norm(X - y, axis=1)
        w = 1.0 / np.maximum(d, eps)
        y_new = (w[:, None] * X).sum(axis=0) / w.sum()
        if np.linalg.norm(y_new - y) < eps:
            return y_new
        y = y_new
    return y


class _BaseSAE:
    """Shared machinery: init, Adam loop, decoder unit-norm + gradient projection."""

    arch = "relu"

    def __init__(self, dim: int, expansion: int = 8, n_features: int | None = None, seed: int = 0):
        self.dim = int(dim)
        self.n_features = int(n_features) if n_features else self.dim * expansion
        rng = np.random.default_rng(seed)
        f, d = self.n_features, self.dim
        self.W_enc = rng.standard_normal((f, d)).astype(np.float64) * (1.0 / np.sqrt(d))
        self.b_enc = np.zeros(f, dtype=np.float64)
        self.W_dec = self.W_enc.T.copy()  # [d, f]; tied init
        self._normalize_decoder()
        self.b_dec = np.zeros(d, dtype=np.float64)
        # sparsity gate metadata carried into the bundle / native kernel
        self.k = 0
        self.threshold: np.ndarray | None = None
        self._rng = rng

    # ---- decoder bookkeeping --------------------------------------------------
    def _normalize_decoder(self):
        norms = np.linalg.norm(self.W_dec, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        self.W_dec /= norms

    def _project_decoder_grad(self, g: np.ndarray) -> np.ndarray:
        """Remove the radial component of the decoder gradient so the unit-norm constraint
        does not fight the optimizer (Anthropic/Bricken recipe)."""
        dots = (g * self.W_dec).sum(axis=0, keepdims=True)
        return g - self.W_dec * dots

    # ---- forward --------------------------------------------------------------
    def _pre(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.W_enc.T + self.b_enc

    def _gate(self, h: np.ndarray) -> np.ndarray:
        """Apply the architecture's sparsity gate to ReLU activations. Overridden."""
        return h

    def encode(self, X: np.ndarray) -> np.ndarray:
        return self._gate(_relu(self._pre(X)))

    def decode(self, H: np.ndarray) -> np.ndarray:
        return np.asarray(H, dtype=np.float64) @ self.W_dec.T + self.b_dec

    def dec_norm2(self) -> np.ndarray:
        return (self.W_dec ** 2).sum(axis=0)

    def quality(self, X: np.ndarray, k: int = 10) -> dict:
        return sae_quality(self, X, k=k)

    # ---- training -------------------------------------------------------------
    def _params(self) -> list[str]:
        return ["W_enc", "b_enc", "W_dec", "b_dec"]

    def _grads(self, xb: np.ndarray, dead: np.ndarray) -> tuple[float, dict, np.ndarray]:
        """Return (loss, grads, fired_mask[f]) for a minibatch. Overridden per arch."""
        raise NotImplementedError

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 40,
        batch_size: int = 128,
        lr: float = 1e-3,
        dead_after: int = 200,
        verbose: bool = False,
    ) -> "_BaseSAE":
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        assert X.shape[1] == self.dim
        self.b_dec = geometric_median(X)

        params = self._params()
        m = {p: np.zeros_like(getattr(self, p)) for p in params}
        v = {p: np.zeros_like(getattr(self, p)) for p in params}
        b1, b2, eps = 0.9, 0.999, 1e-8
        t = 0
        since_fired = np.zeros(self.n_features, dtype=np.int64)
        rng = np.random.default_rng(0)

        for ep in range(epochs):
            perm = rng.permutation(n)
            last = 0.0
            for s in range(0, n, batch_size):
                xb = X[perm[s : s + batch_size]]
                dead = since_fired > dead_after
                loss, grads, fired = self._grads(xb, dead)
                last = loss
                if "W_dec" in grads:
                    grads["W_dec"] = self._project_decoder_grad(grads["W_dec"])
                t += 1
                for p in params:
                    g = grads[p]
                    m[p] = b1 * m[p] + (1 - b1) * g
                    v[p] = b2 * v[p] + (1 - b2) * g ** 2
                    mhat = m[p] / (1 - b1 ** t)
                    vhat = v[p] / (1 - b2 ** t)
                    setattr(self, p, getattr(self, p) - lr * mhat / (np.sqrt(vhat) + eps))
                self._normalize_decoder()
                since_fired = np.where(fired, 0, since_fired + 1)
            if verbose:
                print(f"[{self.arch}] epoch {ep + 1}/{epochs}  loss={last:.5f}")
        self._finalize(X)
        return self

    def _finalize(self, X: np.ndarray):
        """Hook for archs that derive a fixed inference gate from the trained state."""
        pass


class ReLUSAE(_BaseSAE):
    """Plain ReLU + L1 sparse autoencoder (the original, zero-knob fallback)."""

    arch = "relu"

    def __init__(self, dim: int, expansion: int = 8, l1: float = 1e-3, seed: int = 0, **kw):
        super().__init__(dim, expansion=expansion, seed=seed, **kw)
        self.l1 = l1

    def _grads(self, xb, dead):
        nb = xb.shape[0]
        pre = xb @ self.W_enc.T + self.b_enc
        h = _relu(pre)
        xhat = h @ self.W_dec.T + self.b_dec
        err = xhat - xb
        loss = float((err ** 2).mean() + self.l1 * np.abs(h).sum(axis=1).mean())

        g_xhat = (2.0 / nb) * err
        gW_dec = g_xhat.T @ h
        gb_dec = g_xhat.sum(axis=0)
        dh = g_xhat @ self.W_dec + (self.l1 / nb) * (h > 0)
        dpre = dh * (pre > 0)
        gW_enc = dpre.T @ xb
        gb_enc = dpre.sum(axis=0)
        grads = {"W_enc": gW_enc, "b_enc": gb_enc, "W_dec": gW_dec, "b_dec": gb_dec}
        return loss, grads, (h > 0).any(axis=0)


def _topk_mask(h: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask keeping the k largest *positive* entries per row."""
    if k <= 0 or k >= h.shape[1]:
        return h > 0
    mask = np.zeros_like(h, dtype=bool)
    idx = np.argpartition(h, -k, axis=1)[:, -k:]
    rows = np.arange(h.shape[0])[:, None]
    mask[rows, idx] = True
    return mask & (h > 0)


class TopKSAE(_BaseSAE):
    """TopK SAE with AuxK dead-feature revival. ``k`` directly sets L0."""

    arch = "topk"

    def __init__(
        self,
        dim: int,
        expansion: int = 8,
        k: int = 32,
        k_aux: int = 512,
        aux_coef: float = 1.0 / 32.0,
        seed: int = 0,
        **kw,
    ):
        super().__init__(dim, expansion=expansion, seed=seed, **kw)
        self.k = int(min(k, self.n_features))
        self.k_aux = int(k_aux)
        self.aux_coef = float(aux_coef)

    def _gate(self, h):
        return h * _topk_mask(h, self.k)

    def _grads(self, xb, dead):
        nb = xb.shape[0]
        pre = xb @ self.W_enc.T + self.b_enc
        h = _relu(pre)
        mask = _topk_mask(h, self.k)
        a = h * mask
        xhat = a @ self.W_dec.T + self.b_dec
        err = xhat - xb
        loss = float((err ** 2).mean())

        g_xhat = (2.0 / nb) * err
        gW_dec = g_xhat.T @ a
        gb_dec = g_xhat.sum(axis=0)
        da = g_xhat @ self.W_dec
        dpre = da * mask * (pre > 0)

        # AuxK: reconstruct the residual e = x - x̂ from top-k_aux *dead* latents; the
        # gradient it produces flows into dead features and revives them.
        if dead.any() and self.aux_coef > 0:
            k_aux = min(self.k_aux, int(dead.sum()))
            h_dead = np.where(dead[None, :], h, 0.0)
            amask = _topk_mask(h_dead, k_aux)
            a_aux = h_dead * amask
            e = xb - xhat  # stop-grad target
            e_hat = a_aux @ self.W_dec.T
            eaux = e_hat - e
            loss += self.aux_coef * float((eaux ** 2).mean())
            g_eaux = self.aux_coef * (2.0 / nb) * eaux
            gW_dec = gW_dec + g_eaux.T @ a_aux
            da_aux = g_eaux @ self.W_dec
            dpre = dpre + da_aux * amask * (pre > 0)

        gW_enc = dpre.T @ xb
        gb_enc = dpre.sum(axis=0)
        grads = {"W_enc": gW_enc, "b_enc": gb_enc, "W_dec": gW_dec, "b_dec": gb_dec}
        return loss, grads, mask.any(axis=0)


class BatchTopKSAE(TopKSAE):
    """BatchTopK: top-(k·batch) across the whole batch during training; a single threshold
    (calibrated at the end) reproduces the sparsity as a feed-forward gate at inference."""

    arch = "batchtopk"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._orig_k = self.k

    def _grads(self, xb, dead):
        nb = xb.shape[0]
        pre = xb @ self.W_enc.T + self.b_enc
        h = _relu(pre)
        # keep the k*nb largest activations across the whole batch
        budget = self.k * nb
        flat = h.ravel()
        mask = np.zeros_like(flat, dtype=bool)
        pos = np.flatnonzero(flat > 0)
        if pos.size > budget:
            keep = pos[np.argpartition(flat[pos], -budget)[-budget:]]
        else:
            keep = pos
        mask[keep] = True
        mask = mask.reshape(h.shape)
        a = h * mask
        xhat = a @ self.W_dec.T + self.b_dec
        err = xhat - xb
        loss = float((err ** 2).mean())
        g_xhat = (2.0 / nb) * err
        gW_dec = g_xhat.T @ a
        gb_dec = g_xhat.sum(axis=0)
        da = g_xhat @ self.W_dec
        dpre = da * mask * (pre > 0)
        gW_enc = dpre.T @ xb
        gb_enc = dpre.sum(axis=0)
        grads = {"W_enc": gW_enc, "b_enc": gb_enc, "W_dec": gW_dec, "b_dec": gb_dec}
        return loss, grads, mask.any(axis=0)

    def _finalize(self, X: np.ndarray):
        # Calibrate a single global threshold so the mean active latents per row ≈ k, then
        # switch inference from top-k to that fixed threshold gate.
        h = _relu(self._pre(X))
        q = 1.0 - (self._orig_k / self.n_features)
        thr = float(np.quantile(h, np.clip(q, 0.0, 1.0)))
        self.threshold = np.full(self.n_features, thr, dtype=np.float64)
        self.k = 0  # inference now uses the threshold gate

    def _gate(self, h):
        if self.threshold is not None:
            return h * (h > self.threshold)
        return h * _topk_mask(h, self.k)


class JumpReLUSAE(_BaseSAE):
    """JumpReLU SAE: a learnable per-feature Heaviside gate θ, trained with straight-through
    estimators and a rectangular pseudo-derivative (Rajamanoharan et al., 2024)."""

    arch = "jumprelu"

    def __init__(
        self,
        dim: int,
        expansion: int = 8,
        sparsity: float = 1e-3,
        bandwidth: float = 1e-3,
        theta_init: float = 1e-3,
        seed: int = 0,
        **kw,
    ):
        super().__init__(dim, expansion=expansion, seed=seed, **kw)
        self.sparsity = float(sparsity)
        self.bandwidth = float(bandwidth)
        self.log_theta = np.full(self.n_features, np.log(theta_init), dtype=np.float64)

    @property
    def threshold(self):  # exposed as the inference gate
        return np.exp(self.log_theta)

    @threshold.setter
    def threshold(self, _):  # base __init__ sets self.threshold=None; ignore
        pass

    def _params(self):
        return ["W_enc", "b_enc", "W_dec", "b_dec", "log_theta"]

    def _gate(self, h):
        return h * (h > self.threshold)

    def _grads(self, xb, dead):
        nb = xb.shape[0]
        theta = self.threshold
        eps = self.bandwidth
        pre = xb @ self.W_enc.T + self.b_enc
        h = _relu(pre)
        gate = (h > theta).astype(np.float64)
        a = h * gate
        xhat = a @ self.W_dec.T + self.b_dec
        err = xhat - xb
        l0_pen = self.sparsity * gate.sum(axis=1).mean()
        loss = float((err ** 2).mean() + l0_pen)

        # rectangular STE kernel K((h-θ)/ε)/ε; d/dθ H(h-θ) ≈ -K/ε
        within = (np.abs(h - theta) < 0.5 * eps).astype(np.float64) / eps

        g_xhat = (2.0 / nb) * err
        gW_dec = g_xhat.T @ a
        gb_dec = g_xhat.sum(axis=0)
        da = g_xhat @ self.W_dec
        # magnitude path (gate treated as constant via STE)
        dpre = da * gate * (pre > 0)
        gW_enc = dpre.T @ xb
        gb_enc = dpre.sum(axis=0)
        # threshold gets gradient from both the recon (via -h·K/ε) and the L0 penalty (+K/ε)
        g_theta = (-(g_xhat @ self.W_dec) * h * within).sum(axis=0)
        g_theta += self.sparsity * (-within).sum(axis=0) / nb
        g_log_theta = g_theta * theta  # chain rule θ = exp(log_theta)
        grads = {
            "W_enc": gW_enc,
            "b_enc": gb_enc,
            "W_dec": gW_dec,
            "b_dec": gb_dec,
            "log_theta": g_log_theta,
        }
        return loss, grads, (a > 0).any(axis=0)


# Backward-compatible alias — the original public name.
SAE = ReLUSAE

_ARCHES = {
    "relu": ReLUSAE,
    "topk": TopKSAE,
    "batchtopk": BatchTopKSAE,
    "jumprelu": JumpReLUSAE,
}
