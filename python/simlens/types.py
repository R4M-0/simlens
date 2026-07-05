"""Typed result objects mirroring simlens-core's types."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Contribution:
    id: str
    name: str | None
    value: float
    confidence: float | None
    polarity: str  # "shared" | "query_only" | "candidate_only" | "neither"

    @property
    def label(self) -> str:
        return self.name or self.id

    @classmethod
    def from_dict(cls, d: dict) -> "Contribution":
        return cls(
            id=d["id"],
            name=d.get("name"),
            value=float(d["value"]),
            confidence=d.get("confidence"),
            polarity=d.get("polarity", "shared"),
        )


@dataclass(frozen=True)
class Attribution:
    score: float
    metric: str
    level: str
    contributions: list[Contribution]
    completeness_residual: float
    coverage: float
    bundle_hash: str | None = None
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Attribution":
        return cls(
            score=float(d["score"]),
            metric=d["metric"],
            level=d["level"],
            contributions=[Contribution.from_dict(c) for c in d["contributions"]],
            completeness_residual=float(d["completeness_residual"]),
            coverage=float(d["coverage"]),
            bundle_hash=d.get("bundle_hash"),
            warnings=list(d.get("warnings", [])),
        )

    def as_sentence(self, max_terms: int = 3) -> str:
        """Deterministic, faithful natural-language rendering (no LLM)."""
        if not self.contributions:
            return f"Score {self.score:.3f}: no contributions above threshold."
        total = sum(abs(c.value) for c in self.contributions) or 1.0
        parts = []
        for c in self.contributions[:max_terms]:
            pct = 100.0 * abs(c.value) / total
            sign = "" if c.value >= 0 else "−"
            parts.append(f"'{c.label}' ({sign}{pct:.0f}%)")
        lead = "Matched mainly on " if self.metric != "euclidean" else "Differed mainly on "
        sentence = lead + ", ".join(parts)
        if self.completeness_residual > 0.05 * (abs(self.score) or 1.0):
            sentence += f" [approx: {self.completeness_residual:.3f} residual]"
        return sentence + "."

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "metric": self.metric,
            "level": self.level,
            "completeness_residual": self.completeness_residual,
            "coverage": self.coverage,
            "bundle_hash": self.bundle_hash,
            "warnings": self.warnings,
            "contributions": [
                {
                    "id": c.id,
                    "name": c.name,
                    "value": c.value,
                    "confidence": c.confidence,
                    "polarity": c.polarity,
                }
                for c in self.contributions
            ],
        }

    def __repr__(self) -> str:
        top = ", ".join(f"{c.label}={c.value:+.3f}" for c in self.contributions[:3])
        return (
            f"Attribution(level={self.level!r}, score={self.score:.3f}, "
            f"residual={self.completeness_residual:.2e}, coverage={self.coverage:.0%}, "
            f"[{top}])"
        )
