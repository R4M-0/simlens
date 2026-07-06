"""Regulated-domains extension: signed, hashed, reproducible decision records.

Wrap any similarity-driven decision into an auditable log entry carrying the explanation,
the bundle provenance hash, an optional HMAC signature, and the completeness residual.
Business-logic-agnostic: you supply the decision and any context.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone

from ..explain import Explainer


class AuditLog:
    def __init__(self, bundle_or_explainer, secret: str | None = None, path: str | None = None,
                 level: str | None = None, top_k: int = 5):
        self.ex = bundle_or_explainer if isinstance(bundle_or_explainer, Explainer) else Explainer(bundle_or_explainer)
        self.secret = secret
        self.path = path
        self.level = level or self.ex.preferred_level()
        self.top_k = top_k

    def record(self, query, candidate, decision=None, context: dict | None = None,
               level: str | None = None) -> dict:
        attr = self.ex.explain(query, candidate, level=level or self.level, top_k=self.top_k)
        rec = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "context": context or {},
            "explanation": attr.to_dict(),
            "bundle_hash": attr.bundle_hash,
            "completeness_residual": attr.completeness_residual,
        }
        rec["record_hash"] = "sha256:" + hashlib.sha256(
            json.dumps(rec, sort_keys=True, default=str).encode()
        ).hexdigest()
        if self.secret:
            rec["signature"] = "hmac:" + hmac.new(
                self.secret.encode(), rec["record_hash"].encode(), hashlib.sha256
            ).hexdigest()
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        return rec

    def verify(self, rec: dict) -> bool:
        """Re-derive the signature to confirm a record hasn't been tampered with."""
        if not self.secret or "signature" not in rec:
            return False
        expect = hmac.new(self.secret.encode(), rec["record_hash"].encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(rec["signature"][len("hmac:"):], expect)

    @staticmethod
    def report(records: list[dict]) -> str:
        """Human-readable 'reason for decision' report (markdown)."""
        lines = ["# Decision report", ""]
        for r in records:
            a = r["explanation"]
            reasons = ", ".join(
                f"{c.get('name') or c['id']} ({c['value']:+.3f})" for c in a["contributions"][:3]
            )
            lines += [
                f"## {r['timestamp']} — decision: {r.get('decision')}",
                f"- score: {a['score']:.3f}  (level={a['level']}, residual={a['completeness_residual']:.3f})",
                f"- reasons: {reasons}",
                f"- provenance: {r.get('bundle_hash')}",
                "",
            ]
        return "\n".join(lines)
