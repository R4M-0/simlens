"""Feature naming: turn a feature's exemplar items into a human label.

Provider-agnostic. `KeywordNamer` needs no dependencies; `LLMNamer` wraps *any*
`complete(prompt)->str` callable, and `from_provider(...)` builds one for a named provider
(openai, gemini, ...) as an optional extra. LLM-proposed names are tagged source="ai" and
remain user-editable via `Bundle.rename_feature`.
"""
from __future__ import annotations

from .keyword import KeywordNamer
from .llm import LLMNamer, from_callable, from_provider

__all__ = ["KeywordNamer", "LLMNamer", "from_callable", "from_provider"]
