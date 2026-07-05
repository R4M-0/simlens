"""A compact, dependency-light (numpy-only) sparse autoencoder trainer.

Not a research-scale trainer — it exists so the package is self-contained and testable.
For production dictionaries, train with a dedicated stack and import via
`simlens.train.import_sae`.
"""
from __future__ import annotations

import numpy as np


def _relu(x):
    return np.maximum(x, 0.0)


class SAE:
    """ReLU sparse autoencoder: h = relu(x @ Wenc.T + benc); x̂ = h @ Wdec.T + bdec."""

    def __init__(self, dim: int, expansion: int = 8, l1: float = 1e-3, seed: int = 0):
        self.dim = dim
        self.n_features = dim * expansion
        self.l1 = l1
        rng = np.random.default_rng(seed)
        f, d = self.n_features, dim
        self.W_enc = rng.standard_normal((f, d)).astype(np.float64) * (1.0 / np.sqrt(d))
        self.b_enc = np.zeros(f, dtype=np.float64)
        # tie decoder to encoder at init, unit-norm columns
        self.W_dec = self.W_enc.T.copy()
        self._normalize_decoder()
        self.b_dec = np.zeros(d, dtype=np.float64)

    def _normalize_decoder(self):
        norms = np.linalg.norm(self.W_dec, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        self.W_dec /= norms

    def encode(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return _relu(X @ self.W_enc.T + self.b_enc)

    def decode(self, H: np.ndarray) -> np.ndarray:
        return H @ self.W_dec.T + self.b_dec

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 40,
        batch_size: int = 128,
        lr: float = 1e-3,
        verbose: bool = False,
    ) -> "SAE":
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        assert X.shape[1] == self.dim
        self.b_dec = X.mean(axis=0)

        # Adam state
        params = ["W_enc", "b_enc", "W_dec", "b_dec"]
        m = {p: np.zeros_like(getattr(self, p)) for p in params}
        v = {p: np.zeros_like(getattr(self, p)) for p in params}
        b1, b2, eps = 0.9, 0.999, 1e-8
        t = 0
        rng = np.random.default_rng(0)

        for ep in range(epochs):
            perm = rng.permutation(n)
            last = 0.0
            for s in range(0, n, batch_size):
                idx = perm[s : s + batch_size]
                xb = X[idx]
                nb = xb.shape[0]
                pre = xb @ self.W_enc.T + self.b_enc
                h = _relu(pre)
                xhat = h @ self.W_dec.T + self.b_dec
                err = xhat - xb

                mse = (err ** 2).mean()
                l1pen = self.l1 * np.abs(h).sum(axis=1).mean()
                last = mse + l1pen

                g_xhat = (2.0 / nb) * err
                gW_dec = g_xhat.T @ h
                gb_dec = g_xhat.sum(axis=0)
                dh = g_xhat @ self.W_dec + (self.l1 / nb) * (h > 0)
                dpre = dh * (pre > 0)
                gW_enc = dpre.T @ xb
                gb_enc = dpre.sum(axis=0)

                grads = {"W_enc": gW_enc, "b_enc": gb_enc, "W_dec": gW_dec, "b_dec": gb_dec}
                t += 1
                for p in params:
                    m[p] = b1 * m[p] + (1 - b1) * grads[p]
                    v[p] = b2 * v[p] + (1 - b2) * grads[p] ** 2
                    mhat = m[p] / (1 - b1 ** t)
                    vhat = v[p] / (1 - b2 ** t)
                    setattr(self, p, getattr(self, p) - lr * mhat / (np.sqrt(vhat) + eps))
                self._normalize_decoder()
            if verbose:
                print(f"epoch {ep + 1}/{epochs}  loss={last:.5f}")
        return self

    def dec_norm2(self) -> np.ndarray:
        return (self.W_dec ** 2).sum(axis=0)
