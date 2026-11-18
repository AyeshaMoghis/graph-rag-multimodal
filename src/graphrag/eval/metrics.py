"""Evaluation metrics."""

from __future__ import annotations

import math
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


# ── Retrieval metrics over relevance groups (Task 3.3) ──────────────────────
# ``relevant`` maps a group id to the chunk ids in that group. A group counts
# once however many of its identical copies are retrieved.


def _first_hits(retrieved: Sequence[str], relevant: dict[str, list[str]], k: int) -> list[bool]:
    """For each of the top-k positions, True if it is the first hit of a relevant group."""
    id_to_group = {cid: g for g, ids in relevant.items() for cid in ids}
    seen: set[str] = set()
    flags = []
    for cid in retrieved[:k]:
        g = id_to_group.get(cid)
        flags.append(g is not None and g not in seen)
        if g is not None:
            seen.add(g)
    return flags


def recall_at_k(retrieved: Sequence[str], relevant: dict[str, list[str]], k: int) -> float:
    if not relevant:
        raise ValueError("no relevant groups for this question")
    return sum(_first_hits(retrieved, relevant, k)) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: dict[str, list[str]]) -> float:
    for rank, hit in enumerate(_first_hits(retrieved, relevant, len(retrieved)), start=1):
        if hit:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: dict[str, list[str]], k: int) -> float:
    """Binary-gain nDCG; the ideal ranking puts every relevant group first."""
    if not relevant:
        raise ValueError("no relevant groups for this question")
    dcg = sum(
        1.0 / math.log2(i + 2) for i, hit in enumerate(_first_hits(retrieved, relevant, k)) if hit
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal
