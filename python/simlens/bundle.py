"""The `.simlens` concept bundle: portable, hashable artifacts for concept-level
attribution (SAE encoder + decoder norms + named features + concept directions)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BUNDLE_VERSION = "1.0"


def _sha256(*chunks: bytes) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return "sha256:" + h.hexdigest()


@dataclass
class Bundle:
    """One embedding space's interpretability artifacts."""

    embedder: str
    dim: int
    metric: str = "cosine"
    modality: str = "text"

    # SAE (optional) — full encoder + decoder (feature-major)
    w_enc: np.ndarray | None = None            # [n_features, dim]
    b_enc: np.ndarray | None = None            # [n_features]
    w_dec: np.ndarray | None = None            # [n_features, dim] (decoder atoms d_f)
    b_dec: np.ndarray | None = None            # [dim]
    sae_k: int = 0                             # top-k sparsity gate (0 = plain ReLU)
    sae_threshold: np.ndarray | None = None    # [n_features] JumpReLU gate (or None)
    feature_names: list[str | None] = field(default_factory=list)
    feature_conf: list[float | None] = field(default_factory=list)
    feature_source: list[str | None] = field(default_factory=list)  # payload|keyword|ai|manual
    feature_exemplars: dict = field(default_factory=dict)  # feat index (str) -> [items]

    # Concepts (optional) — unit CAV directions
    concept_names: list[str] = field(default_factory=list)
    concept_dirs: np.ndarray | None = None     # [n_concepts, dim]
    concept_conf: list[float] = field(default_factory=list)
    concept_source: list[str] = field(default_factory=list)  # examples | text
    concept_exemplars: dict = field(default_factory=dict)  # concept name -> [items]
    aspects: dict[str, str] = field(default_factory=dict)  # concept -> aspect bucket

    # Anisotropy correction (T2.2) — the centered/whitened "why" view
    mean: np.ndarray | None = None             # [dim] corpus centroid μ
    whitening: np.ndarray | None = None        # [dim, dim] linear map (ABTT / whitening)

    # Faithfulness certification (T2.3) — a signed quality scorecard
    faithfulness: dict = field(default_factory=dict)

    content_hash: str | None = None
    signature: str | None = None

    # ---- properties -----------------------------------------------------------
    @property
    def n_features(self) -> int:
        return 0 if self.w_enc is None else int(self.w_enc.shape[0])

    @property
    def has_sae(self) -> bool:
        return self.w_enc is not None

    @property
    def has_concepts(self) -> bool:
        return self.concept_dirs is not None and len(self.concept_names) > 0

    # ---- hashing / provenance -------------------------------------------------
    def compute_hash(self) -> str:
        chunks: list[bytes] = [self.embedder.encode(), str(self.dim).encode(), self.metric.encode()]
        for arr in (
            self.w_enc, self.b_enc, self.w_dec, self.b_dec, self.concept_dirs,
            self.sae_threshold, self.mean, self.whitening,
        ):
            if arr is not None:
                chunks.append(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
        chunks.append(str(int(self.sae_k)).encode())
        chunks.append(json.dumps(self.concept_names, sort_keys=True).encode())
        chunks.append(json.dumps(self.faithfulness, sort_keys=True).encode())
        return _sha256(*chunks)

    def _manifest(self) -> dict:
        sae = None
        if self.has_sae:
            gate = "relu"
            if self.sae_threshold is not None:
                gate = "jumprelu"
            elif self.sae_k:
                gate = "topk"
            sae = {"n_features": self.n_features, "gate": gate, "k": int(self.sae_k)}
        return {
            "simlens_bundle_version": BUNDLE_VERSION,
            "embedder": {"id": self.embedder, "dim": self.dim, "modality": self.modality},
            "metric": self.metric,
            "sae": sae,
            "concepts": len(self.concept_names),
            "centered": self.mean is not None,
            "faithfulness": self.faithfulness or None,
            "content_hash": self.content_hash,
            "signature": self.signature,
        }

    # ---- IO -------------------------------------------------------------------
    def save(self, path: str | Path) -> "Bundle":
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.content_hash = self.compute_hash()

        arrays = {}
        if self.has_sae:
            arrays.update(
                w_enc=np.asarray(self.w_enc, dtype=np.float32),
                b_enc=np.asarray(self.b_enc, dtype=np.float32),
                w_dec=np.asarray(self.w_dec, dtype=np.float32),
                b_dec=np.asarray(self.b_dec, dtype=np.float32),
            )
            if self.sae_threshold is not None:
                arrays["sae_threshold"] = np.asarray(self.sae_threshold, dtype=np.float32)
        if self.has_concepts:
            arrays["concept_dirs"] = np.asarray(self.concept_dirs, dtype=np.float32)
        if self.mean is not None:
            arrays["mean"] = np.asarray(self.mean, dtype=np.float32)
        if self.whitening is not None:
            arrays["whitening"] = np.asarray(self.whitening, dtype=np.float32)
        if arrays:
            np.savez(path / "weights.npz", **arrays)

        (path / "manifest.json").write_text(json.dumps(self._manifest(), indent=2))
        (path / "features.json").write_text(
            json.dumps(
                {
                    "names": self.feature_names,
                    "conf": self.feature_conf,
                    "source": self.feature_source,
                    "exemplars": self.feature_exemplars,
                }
            )
        )
        (path / "concepts.json").write_text(
            json.dumps(
                {
                    "names": self.concept_names,
                    "conf": self.concept_conf,
                    "source": self.concept_source,
                    "aspects": self.aspects,
                    "exemplars": self.concept_exemplars,
                }
            )
        )
        return self

    def save_rust(self, path: str | Path) -> "Bundle":
        """Also emit a Rust-loadable form (`manifest.json` + `weights.safetensors`) for the
        native loader (`simlens_core::bundle`, feature `bundle`). Needs `safetensors`."""
        try:
            from safetensors.numpy import save_file
        except ImportError as e:  # pragma: no cover
            raise ImportError("save_rust needs safetensors; `pip install simlens[train]`") from e
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.content_hash = self.compute_hash()
        tensors = {}
        if self.has_sae:
            tensors["w_enc"] = np.ascontiguousarray(self.w_enc, np.float32)
            tensors["b_enc"] = np.ascontiguousarray(self.b_enc, np.float32)
            tensors["w_dec"] = np.ascontiguousarray(self.w_dec, np.float32)
            tensors["b_dec"] = np.ascontiguousarray(self.b_dec, np.float32)
            if self.sae_threshold is not None:
                tensors["sae_threshold"] = np.ascontiguousarray(self.sae_threshold, np.float32)
        if self.has_concepts:
            tensors["concept_dirs"] = np.ascontiguousarray(self.concept_dirs, np.float32)
        if tensors:
            save_file(tensors, str(path / "weights.safetensors"))
        (path / "manifest.json").write_text(json.dumps(self._manifest(), indent=2))
        return self

    @classmethod
    def load(cls, path: str | Path) -> "Bundle":
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())
        emb = manifest["embedder"]
        b = cls(
            embedder=emb["id"],
            dim=int(emb["dim"]),
            metric=manifest.get("metric", "cosine"),
            modality=emb.get("modality", "text"),
            content_hash=manifest.get("content_hash"),
        )
        sae_meta = manifest.get("sae") or {}
        b.sae_k = int(sae_meta.get("k", 0))
        b.faithfulness = manifest.get("faithfulness") or {}
        wpath = path / "weights.npz"
        if wpath.exists():
            w = np.load(wpath)
            if "w_enc" in w:
                b.w_enc = w["w_enc"]
                b.b_enc = w["b_enc"]
                b.w_dec = w["w_dec"]
                b.b_dec = w["b_dec"]
            if "sae_threshold" in w:
                b.sae_threshold = w["sae_threshold"]
            if "concept_dirs" in w:
                b.concept_dirs = w["concept_dirs"]
            if "mean" in w:
                b.mean = w["mean"]
            if "whitening" in w:
                b.whitening = w["whitening"]
        b.signature = manifest.get("signature")
        fp = path / "features.json"
        if fp.exists():
            f = json.loads(fp.read_text())
            b.feature_names = f.get("names", [])
            b.feature_conf = f.get("conf", [])
            b.feature_source = f.get("source", [])
            b.feature_exemplars = f.get("exemplars", {})
        cp = path / "concepts.json"
        if cp.exists():
            c = json.loads(cp.read_text())
            b.concept_names = c.get("names", [])
            b.concept_conf = c.get("conf", [])
            b.concept_source = c.get("source", [])
            b.aspects = c.get("aspects", {})
            b.concept_exemplars = c.get("exemplars", {})
        return b

    def verify(self) -> bool:
        """True iff the on-disk hash matches a fresh recomputation."""
        return self.content_hash == self.compute_hash()

    def sign(self, secret: str) -> "Bundle":
        """Attach an HMAC-SHA256 signature over the content hash (audit mode, §13.8)."""
        import hmac

        self.content_hash = self.compute_hash()
        self.signature = "hmac:" + hmac.new(
            secret.encode(), self.content_hash.encode(), hashlib.sha256
        ).hexdigest()
        return self

    def verify_signature(self, secret: str) -> bool:
        """True iff the signature matches `secret` over the current content hash."""
        import hmac

        if not self.signature or not self.signature.startswith("hmac:"):
            return False
        expect = hmac.new(
            secret.encode(), self.compute_hash().encode(), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(self.signature[len("hmac:") :], expect)

    def add_concept(
        self,
        name: str,
        positive: np.ndarray,
        negative: np.ndarray,
        aspect: str | None = None,
    ) -> "Bundle":
        """Fit and register a named concept direction from example sets."""
        from .train.cav import fit_cav

        direction, acc = fit_cav(positive, negative)
        if self.concept_dirs is None:
            self.concept_dirs = direction.reshape(1, -1)
        else:
            self.concept_dirs = np.vstack([self.concept_dirs, direction.reshape(1, -1)])
        self.concept_names.append(name)
        self.concept_conf.append(acc)
        self.concept_source.append("examples")
        if aspect:
            self.aspects[name] = aspect
        return self

    def add_text_concept(self, name: str, direction: np.ndarray, aspect: str | None = None) -> "Bundle":
        """Register a zero-shot concept from an already-embedded phrase direction (§Part A).

        `direction` is a raw (dim,) embedding of the concept phrase; it is unit-normalized
        here. Confidence is 0.0 (unvalidated) until calibrated against examples.
        """
        d = np.asarray(direction, dtype=np.float32).ravel()
        n = np.linalg.norm(d)
        if n > 0:
            d = d / n
        if self.concept_dirs is None:
            self.concept_dirs = d.reshape(1, -1)
        else:
            self.concept_dirs = np.vstack([self.concept_dirs, d.reshape(1, -1)])
        self.concept_names.append(name)
        self.concept_conf.append(0.0)
        self.concept_source.append("text")
        if aspect:
            self.aspects[name] = aspect
        return self

    def add_cross_modal_concept(
        self,
        name: str,
        direction: np.ndarray,
        source_modality: str = "text",
    ) -> "Bundle":
        """Register a concept defined in one modality's encoder and applied in a shared,
        cross-modal space (CLIP-style): e.g. a *text* direction used to explain *image*
        embeddings. Clearly tagged so consumers know it is semantic/cross-modal, not a CAV
        fit on in-modality examples.
        """
        d = np.asarray(direction, dtype=np.float32).ravel()
        n = np.linalg.norm(d)
        if n > 0:
            d = d / n
        if self.concept_dirs is None:
            self.concept_dirs = d.reshape(1, -1)
        else:
            self.concept_dirs = np.vstack([self.concept_dirs, d.reshape(1, -1)])
        self.concept_names.append(name)
        self.concept_conf.append(0.0)  # unvalidated in-modality; it's a semantic transfer
        self.concept_source.append("cross_modal")
        self.aspects[name] = f"cross-modal:{source_modality}→{self.modality}"
        return self

    def rename_feature(self, index: int, name: str, source: str = "manual") -> "Bundle":
        """Manually (re)name a feature — user override of auto/AI-proposed names."""
        if index >= len(self.feature_names):
            raise IndexError(f"feature {index} out of range")
        self.feature_names[index] = name
        if index < len(self.feature_source):
            self.feature_source[index] = source
        return self

    def name_provenance(self) -> dict:
        """Count named features by provenance (payload / keyword / ai / manual)."""
        from collections import Counter

        return dict(Counter(s for s in self.feature_source if s))

    def certify(self, vectors, **kw) -> dict:
        """Compute and attach a faithfulness scorecard (T2.3), covered by the content hash."""
        from .eval.certify import certify as _certify

        card = _certify(self, vectors, **kw)
        self.content_hash = self.compute_hash()
        return card

    # ---- native bridges -------------------------------------------------------
    def to_native_sae(self):
        from ._native import PySae

        n = self.n_features
        names = list(self.feature_names) + [None] * (n - len(self.feature_names))
        conf = list(self.feature_conf) + [None] * (n - len(self.feature_conf))
        threshold = (
            np.ascontiguousarray(self.sae_threshold, dtype=np.float32)
            if self.sae_threshold is not None
            else None
        )
        # zero-copy construction straight from the numpy weight buffers (T2.4)
        sae = PySae.from_numpy(
            self.dim,
            n,
            np.ascontiguousarray(self.w_enc, dtype=np.float32),
            np.ascontiguousarray(self.b_enc, dtype=np.float32),
            np.ascontiguousarray(self.w_dec, dtype=np.float32),
            np.ascontiguousarray(self.b_dec, dtype=np.float32),
            int(self.sae_k),
            threshold,
        )
        sae.set_labels(names[:n], conf[:n])
        return sae

    def to_native_cav(self):
        from ._native import PyCavSet

        return PyCavSet(
            self.dim,
            list(self.concept_names),
            np.ascontiguousarray(self.concept_dirs, dtype=np.float32).tobytes(),
            [float(x) for x in self.concept_conf],
        )
