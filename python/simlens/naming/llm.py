"""LLM-based namer, abstracted over any provider via a `complete(prompt)->str` callable."""
from __future__ import annotations

from typing import Callable

from .base import _exemplar_texts

_PROMPT = (
    "The following items were all found to strongly share a single, specific concept or "
    "property:\n{items}\n\nReply with ONLY a short (2-4 word) human-readable label naming "
    "what they share. No punctuation, no explanation."
)


class LLMNamer:
    """Wraps any text-completion callable. Names are tagged source='ai' (user-editable)."""

    source = "ai"

    def __init__(self, complete: Callable[[str], str], max_exemplars: int = 8, confidence: float = 0.5):
        self.complete = complete
        self.max_exemplars = max_exemplars
        self.confidence = confidence

    def name(self, exemplars: list) -> tuple[str | None, float]:
        texts = _exemplar_texts(exemplars)[: self.max_exemplars]
        if not texts:
            return None, 0.0
        prompt = _PROMPT.format(items="\n".join(f"- {t}" for t in texts))
        try:
            resp = (self.complete(prompt) or "").strip()
        except Exception:  # noqa: BLE001 - naming is best-effort, never fatal
            return None, 0.0
        label = resp.splitlines()[0].strip().strip(".\"'` ") if resp else ""
        return (label or None), (self.confidence if label else 0.0)


def from_callable(complete: Callable[[str], str], **kw) -> LLMNamer:
    """Build an LLMNamer from your own `complete(prompt) -> str` function."""
    return LLMNamer(complete, **kw)


def from_provider(provider: str, model: str | None = None, api_key: str | None = None, **kw) -> LLMNamer:
    """Build an LLMNamer for a named provider (optional extras).

    provider="openai"  → needs `pip install simlens[openai]`
    provider="gemini"  → needs `pip install simlens[gemini]`
    """
    p = provider.lower()
    if p == "openai":
        complete = _openai_complete(model or "gpt-4o-mini", api_key)
    elif p in ("gemini", "google"):
        complete = _gemini_complete(model or "gemini-2.0-flash", api_key)
    else:
        raise ValueError(f"unknown provider {provider!r}; use 'openai', 'gemini', or from_callable")
    return LLMNamer(complete, **kw)


def _openai_complete(model: str, api_key: str | None) -> Callable[[str], str]:
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise ImportError("provider 'openai' needs `pip install simlens[openai]`") from e
    client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def complete(prompt: str) -> str:
        r = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=16
        )
        return r.choices[0].message.content or ""

    return complete


def _gemini_complete(model: str, api_key: str | None) -> Callable[[str], str]:
    try:
        import google.generativeai as genai
    except ImportError as e:  # pragma: no cover
        raise ImportError("provider 'gemini' needs `pip install simlens[gemini]`") from e
    if api_key:
        genai.configure(api_key=api_key)
    gm = genai.GenerativeModel(model)

    def complete(prompt: str) -> str:
        return gm.generate_content(prompt).text or ""

    return complete
