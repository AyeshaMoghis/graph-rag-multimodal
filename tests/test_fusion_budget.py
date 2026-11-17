from __future__ import annotations

import pytest

from graphrag.retrieval.fusion import (
    NO_GRAPH,
    BudgetConfig,
    estimate_tokens,
    naive_conflicts,
    pack_context,
    truncate_to_budget,
)


def test_estimate_tokens() -> None:
    assert estimate_tokens("one two three") == 4  # 3 / 0.75


def test_truncate_stops_at_first_overflow() -> None:
    items = [{"text": "w " * 30}, {"text": "w " * 30}, {"text": "w"}]  # 40, 40, 1 tokens
    kept = truncate_to_budget(items, budget_tokens=60)
    assert kept == items[:1]  # stops, does not skip ahead to the small third item


def test_pack_respects_40_40_20_split() -> None:
    texts = [{"chunk_id": f"c{i}", "text": "w " * 600} for i in range(5)]  # 800 tokens each
    images = [{"caption": "w " * 600} for _ in range(3)]
    packed = pack_context([], [], texts, images, BudgetConfig(total_tokens=6000))
    assert len(packed.text_chunks) == 3  # 2400 budget
    assert len(packed.images) == 1  # 1200 budget
    assert packed.graph_context == NO_GRAPH
    assert packed.chunk_ids == ["c0", "c1", "c2"]


def test_graph_context_truncated_to_budget() -> None:
    paths = [f"A{i} -[USES]-> B{i}" for i in range(2000)]
    packed = pack_context(paths, [], [], [], BudgetConfig(total_tokens=600))
    assert packed.graph_context.endswith("[truncated]")
    assert estimate_tokens(packed.graph_context) <= 240 + 2
    assert packed.metadata()["graph_paths_total"] == 2000


def test_budget_fractions_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        BudgetConfig(graph_fraction=0.5, text_fraction=0.5, image_fraction=0.5)


def test_naive_conflicts_substring_heuristic() -> None:
    chunks = [{"text": "This is not MBTI based."}, {"text": "Neutral text."}]
    assert naive_conflicts(chunks, ["MBTI", "BM25"]) == [
        "Potential conflict: 'mbti' mentioned negatively in text."
    ]
