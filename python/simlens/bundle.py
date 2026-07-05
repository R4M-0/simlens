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

    # SAE (optional) — encoder + precomputed decoder column norms²
    w_enc: np.ndarray | None = None            # [n_features, dim]
    b_enc: np.ndarray | None = None            # [n_features]
    dec_norm2: np.ndarray | None = None        # [n_features]
    feature_names: list[str | None] = field(default_factory=list)
    feature_conf: list[float | None] = field(default_factory=list)

    # Concepts (optional) — unit CAV directions
    concept_names: list[str] = field(default_factory=list)
    concept_dirs: np.ndarray | None = None     # [n_concepts, dim]
    concept_conf: list[float] = field(default_factory=list)
    aspects: dict[str, str] = field(default_factory=dict)  # concept -> aspect bucket

    content_hash: str | None = None

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
        for arr in (self.w_enc, self.b_enc, self.dec_norm2, self.concept_dirs):
            if arr is not None:
                chunks.append(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
        chunks.append(json.dumps(self.concept_names, sort_keys=True).encode())
        return _sha256(*chunks)

    def _manifest(self) -> dict:
        return {
            "simlens_bundle_version": BUNDLE_VERSION,
            "embedder": {"id": self.embedder, "dim": self.dim, "modality": self.modality},
            "metric": self.metric,
            "sae": None if not self.has_sae else {"n_features": self.n_features},
            "concepts": len(self.concept_names),
            "content_hash": self.content_hash,
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
                dec_norm2=np.asarray(self.dec_norm2, dtype=np.float32),
            )
        if self.has_concepts:
            arrays["concept_dirs"] = np.asarray(self.concept_dirs, dtype=np.float32)
        if arrays:
            np.savez(path / "weights.npz", **arrays)

        (path / "manifest.json").write_text(json.dumps(self._manifest(), indent=2))
        (path / "features.json").write_text(
            json.dumps({"names": self.feature_names, "conf": self.feature_conf})
        )
        (path / "concepts.json").write_text(
            json.dumps(
                {
                    "names": self.concept_names,
                    "conf": self.concept_conf,
                    "aspects": self.aspects,
                }
            )
        )
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
        wpath = path / "weights.npz"
        if wpath.exists():
            w = np.load(wpath)
            if "w_enc" in w:
                b.w_enc = w["w_enc"]
                b.b_enc = w["b_enc"]
                b.dec_norm2 = w["dec_norm2"]
            if "concept_dirs" in w:
                b.concept_dirs = w["concept_dirs"]
        fp = path / "features.json"
        if fp.exists():
            f = json.loads(fp.read_text())
            b.feature_names = f.get("names", [])
            b.feature_conf = f.get("conf", [])
        cp = path / "concepts.json"
        if cp.exists():
            c = json.loads(cp.read_text())
            b.concept_names = c.get("names", [])
            b.concept_conf = c.get("conf", [])
            b.aspects = c.get("aspects", {})
        return b

    def verify(self) -> bool:
        """True iff the on-disk hash matches a fresh recomputation."""
        return self.content_hash == self.compute_hash()

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
        if aspect:
            self.aspects[name] = aspect
        return self

    # ---- native bridges -------------------------------------------------------
    def to_native_sae(self):
        from ._native import PySae

        n = self.n_features
        names = list(self.feature_names) + [None] * (n - len(self.feature_names))
        conf = list(self.feature_conf) + [None] * (n - len(self.feature_conf))
        sae = PySae(
            self.dim,
            n,
            np.ascontiguousarray(self.w_enc, dtype=np.float32).tobytes(),
            np.ascontiguousarray(self.b_enc, dtype=np.float32).tobytes(),
            np.ascontiguousarray(self.dec_norm2, dtype=np.float32).tobytes(),
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
