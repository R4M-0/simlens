"""Optional torch/GPU SAE backend (``pip install simlens[train]``).

SAEs need scale, and torch gives autodiff + GPU. To keep the *inference* path identical and
already-tested, we only use torch for the optimization: after training we copy the learned
weights into the matching numpy SAE class, so ``encode``/``decode``/attribution reuse the
exact same code as the numpy backend (train == inference, one gate implementation).
"""
from __future__ import annotations

import numpy as np

from .sae import _ARCHES, geometric_median


def _topk(z, k):
    import torch

    if k <= 0 or k >= z.shape[1]:
        return torch.relu(z)
    vals, idx = torch.topk(torch.relu(z), k, dim=1)
    out = torch.zeros_like(z)
    return out.scatter(1, idx, vals)


def fit_torch(
    X: np.ndarray,
    arch: str = "topk",
    k: int = 32,
    expansion: int = 8,
    epochs: int = 60,
    batch_size: int = 512,
    lr: float = 3e-4,
    seed: int = 0,
    device: str | None = None,
    verbose: bool = False,
    **kw,
):
    import torch

    torch.manual_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.tensor(np.asarray(X, dtype=np.float32), device=dev)
    n, dim = Xt.shape
    nf = dim * expansion

    W_dec = torch.nn.functional.normalize(torch.randn(dim, nf, device=dev), dim=0)
    W_enc = W_dec.t().clone()
    b_enc = torch.zeros(nf, device=dev)
    b_dec = torch.tensor(geometric_median(X), dtype=torch.float32, device=dev)
    for t in (W_enc, b_enc, W_dec, b_dec):
        t.requires_grad_(True)
    params = [W_enc, b_enc, W_dec, b_dec]
    log_theta = None
    if arch == "jumprelu":
        log_theta = torch.full((nf,), np.log(1e-3), device=dev, requires_grad=True)
        params.append(log_theta)
    opt = torch.optim.Adam(params, lr=lr)

    kk = min(k, nf)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        last = 0.0
        for s in range(0, n, batch_size):
            xb = Xt[perm[s : s + batch_size]]
            z = xb @ W_enc.t() + b_enc
            if arch in ("topk", "batchtopk"):
                a = _topk(z, kk)
            elif arch == "jumprelu":
                theta = torch.exp(log_theta)
                h = torch.relu(z)
                a = h * (h > theta).float()  # STE: gate constant in backward
            else:  # relu
                a = torch.relu(z)
            xhat = a @ W_dec.t() + b_dec
            loss = ((xhat - xb) ** 2).mean()
            if arch == "relu":
                loss = loss + kw.get("l1", 1e-3) * a.abs().sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                W_dec.data = torch.nn.functional.normalize(W_dec.data, dim=0)
            last = float(loss)
        if verbose:
            print(f"[torch {arch}] epoch {ep + 1}/{epochs} loss={last:.5f}")

    # hand the learned weights to the numpy class for a single, tested inference path
    cls = _ARCHES[arch]
    ctor = {"dim": dim, "expansion": expansion, "seed": seed}
    if arch in ("topk", "batchtopk"):
        ctor["k"] = kk
    sae = cls(**ctor)
    sae.W_enc = W_enc.detach().cpu().numpy().astype(np.float64)
    sae.b_enc = b_enc.detach().cpu().numpy().astype(np.float64)
    sae.W_dec = W_dec.detach().cpu().numpy().astype(np.float64)
    sae.b_dec = b_dec.detach().cpu().numpy().astype(np.float64)
    if arch == "jumprelu":
        sae.log_theta = log_theta.detach().cpu().numpy().astype(np.float64)
    if arch == "batchtopk":
        sae._finalize(np.asarray(X, dtype=np.float64))
    return sae
