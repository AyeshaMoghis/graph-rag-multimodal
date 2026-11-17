"""Evaluation metrics."""

from __future__ import annotations

from collections.abc import Sequence


def keyword_recall(text: str, expected_concepts: Sequence[str]) -> float:
    """Fraction of expected concepts found case-insensitively in ``text`` (original metric).

    Rounded to 3 places, as the notebook did, so baseline numbers stay comparable.
    """
    if not expected_concepts:
        return 1.0
    lower = text.lower()
    hits = sum(1 for kw in expected_concepts if kw.lower() in lower)
    return round(hits / len(expected_concepts), 3)


def context_keyword_recall(chunks: Sequence[str], expected_concepts: Sequence[str]) -> float:
    """Keyword recall over retrieved context instead of the generated answer.

    Needs no LLM, so it isolates retrieval quality. It is an upper bound on what
    the generator could mention from text retrieval alone.
    """
    return keyword_recall("\n".join(chunks), expected_concepts)
