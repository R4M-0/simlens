"""T1.1b — detection-scored naming. A mock LLM stands in for a provider so we can test the
generation + detection machinery deterministically, including the acceptance criterion:
the measured score correlates with ground-truth correctness on known concepts."""
import re
from collections import Counter

import numpy as np
import pytest

from simlens.naming import (
    ValidatedNamer,
    decile_exemplars,
    negative_exemplars,
    score_detection,
)

_STOP = {"the", "a", "of", "and", "report", "document", "item"}


class MockLLM:
    """A stand-in provider: generation returns the shared keyword; detection selects the
    items that literally contain the label word (a perfect classifier for keyword concepts)."""

    def __call__(self, prompt: str) -> str:
        if "Reply with ONLY a short" in prompt:
            toks = Counter()
            for line in prompt.splitlines():
                if line.startswith("- "):
                    toks.update(w for w in re.findall(r"[a-z]+", line.lower()) if w not in _STOP)
            return toks.most_common(1)[0][0] if toks else "thing"
        if "match that description" in prompt:
            label = re.search(r'described as: "([^"]+)"', prompt).group(1).lower()
            picks = []
            for line in prompt.splitlines():
                m = re.match(r"(\d+)\.\s+(.*)", line)
                if m and label in m.group(2).lower():
                    picks.append(m.group(1))
            return ", ".join(picks) if picks else "none"
        return ""


def test_score_detection_perfect_and_chance():
    llm = MockLLM()
    pos = ["finance market", "finance crash", "finance stocks"]
    neg = ["sports game", "sports team", "sports match"]
    assert score_detection(llm, "finance", pos, neg) == pytest.approx(1.0)
    # a wrong label the classifier can't detect → chance-level balanced accuracy
    assert score_detection(llm, "banana", pos, neg) < 0.6


def test_validated_namer_drops_low_score():
    llm = MockLLM()
    namer = ValidatedNamer(llm, min_score=0.7)
    pos = ["finance market", "finance crash"]
    neg = ["sports game", "sports team"]
    label, acc = namer.name_scored(pos, pos, neg)
    assert label == "finance" and acc >= 0.7
    # if positives don't share a detectable keyword, the name is dropped
    noise = ["alpha one", "beta two", "gamma three"]
    label2, acc2 = namer.name_scored(noise, noise, neg)
    assert label2 is None


def test_decile_and_negative_sampling():
    acts = np.linspace(0, 1, 100)
    items = [f"item{i}" for i in range(100)]
    ex = decile_exemplars(acts, items, n_per_decile=1)
    assert 8 <= len(ex) <= 10  # roughly one per decile
    acts2 = np.array([0.0, 0.0, 0.5, 0.9])
    neg = negative_exemplars(acts2, ["a", "b", "c", "d"], n=2)
    assert set(neg) <= {"a", "b"}  # only non-activating items


def test_autofit_uses_scored_naming_end_to_end():
    import simlens
    from simlens.adapters import Memory

    rng = np.random.default_rng(0)
    dim, n = 24, 150
    topics = ["finance", "sports", "medicine"]
    protos = rng.standard_normal((3, dim))
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    store = Memory()
    for i in range(n):
        t = i % 3
        v = (protos[t] + 0.05 * rng.standard_normal(dim)).astype(np.float32)
        # no payload label field that names features, so the Namer must do the work
        store.add(f"id{i}", v, payload={"text": f"{topics[t]} article number {i}"})
    namer = ValidatedNamer(MockLLM(), min_score=0.5)
    b = simlens.autofit(store, embedder="syn", expansion=4, epochs=20, namer=namer)
    ai = [i for i, s in enumerate(b.feature_source) if s == "ai"]
    assert ai, "expected at least one detection-scored AI name"
    for i in ai:
        assert 0.0 <= b.feature_conf[i] <= 1.0  # confidence is the detection accuracy


def test_detection_score_correlates_with_correctness():
    """Acceptance: on synthetic known concepts, the measured detection score correlates
    (Spearman > 0.6) with whether the name is actually right."""
    llm = MockLLM()
    concepts = ["finance", "sports", "medicine", "weather", "cooking", "travel"]
    correct_flags, scores = [], []
    rng = np.random.default_rng(0)
    for c in concepts:
        pos = [f"{c} story {i}" for i in range(4)]
        neg = [f"{rng.choice([x for x in concepts if x != c])} note {i}" for i in range(4)]
        # correct name (the concept) → high score
        correct_flags.append(1)
        scores.append(score_detection(llm, c, pos, neg))
        # wrong name (a random other concept) → low score
        wrong = str(rng.choice([x for x in concepts if x != c]))
        correct_flags.append(0)
        scores.append(score_detection(llm, wrong, pos, neg))

    # Spearman via ranks (no scipy dependency)
    def spearman(a, b):
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    assert spearman(correct_flags, scores) > 0.6
