"""T1.2 — run the validation notebooks end-to-end (offline hashing fallback) as a CI guard.
Each notebook's main() returns its faithfulness verdict, which must hold."""
import os
import sys

import pytest

_EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")
sys.path.insert(0, os.path.abspath(_EXAMPLES))


@pytest.mark.parametrize("module", ["notebook_rag", "notebook_recsys", "notebook_kg", "notebook_audit"])
def test_notebook_runs_and_is_faithful(module, monkeypatch):
    # force the offline hashing embedder so the guard is deterministic and network-free
    import _corpus

    monkeypatch.setattr(_corpus, "embed",
                        lambda texts, dim=256: (_corpus.hashing_embed(texts, dim), "hashing-bow"))
    mod = __import__(module)
    assert mod.main() is True
