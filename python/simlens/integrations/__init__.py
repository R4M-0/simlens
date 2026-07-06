"""System extensions — thin, customizable wrappers that adapt SimLens to a system type.

Each is business-logic-agnostic: you bring the vectors, it brings the explanation.

    from simlens.integrations.rag import RagExplainer
    from simlens.integrations.recsys import RecsysExplainer
    from simlens.integrations.kg import KnowledgeGraphExplainer
    from simlens.integrations.audit import AuditLog
"""
from __future__ import annotations

from .audit import AuditLog
from .kg import KnowledgeGraphExplainer
from .rag import RagExplainer
from .recsys import RecsysExplainer

__all__ = ["RagExplainer", "RecsysExplainer", "KnowledgeGraphExplainer", "AuditLog"]
