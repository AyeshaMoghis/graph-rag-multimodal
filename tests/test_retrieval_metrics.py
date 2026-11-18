"""Recall@k, MRR, nDCG@k over relevance groups, and relevance-label loading."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from graphrag.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from graphrag.eval.relevance import load_relevance, text_group

REL = {"g1": ["a", "a_copy"], "g2": ["b"]}


def test_recall_counts_each_group_once() -> None:
    assert recall_at_k(["a", "a_copy", "x"], REL, 5) == 0.5  # the copy adds nothing
    assert recall_at_k(["x", "b", "a"], REL, 5) == 1.0
    assert recall_at_k(["x", "y", "b"], REL, 2) == 0.0


def test_mrr() -> None:
    assert reciprocal_rank(["x", "a_copy", "b"], REL) == 0.5
    assert reciprocal_rank(["x", "y"], REL) == 0.0


def test_ndcg() -> None:
    assert ndcg_at_k(["a", "b"], REL, 5) == pytest.approx(1.0)
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["x", "a", "a_copy"], REL, 5) == pytest.approx(expected)


def test_metrics_reject_unlabelled_questions() -> None:
    with pytest.raises(ValueError):
        recall_at_k(["a"], {}, 5)


def test_text_group_ignores_whitespace() -> None:
    assert text_group("Trolls  post\ncontent") == text_group("Trolls post content")


def test_reviewed_labels_override_suggestions(tmp_path: Path) -> None:
    data = {
        "questions": [
            {
                "id": "q1",
                "reviewed": False,
                "candidates": [
                    {"group": "g1", "chunk_ids": ["a"], "suggested": True, "relevant": None},
                    {"group": "g2", "chunk_ids": ["b"], "suggested": False, "relevant": None},
                ],
            },
            {
                "id": "q2",
                "reviewed": True,
                "candidates": [
                    {"group": "g1", "chunk_ids": ["a"], "suggested": True, "relevant": False},
                    {"group": "g2", "chunk_ids": ["b"], "suggested": False, "relevant": True},
                ],
            },
        ]
    }
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(data))
    labels, reviewed = load_relevance(path)
    assert labels == {"q1": {"g1": ["a"]}, "q2": {"g2": ["b"]}}
    assert reviewed == {"q1": False, "q2": True}


def test_render_table_marks_pending_and_provisional(tmp_path: Path) -> None:
    from graphrag.eval.ablation import RowResult, render_table, save_ablation

    measured = RowResult("hybrid", {}, "measured")
    run = {
        "recall_at_5": 0.5,
        "mrr": 0.7,
        "ndcg_at_5": 0.6,
        "context_keyword_recall": 0.73,
        "latency_ms_p50": 200.0,
        "questions_labelled": 50.0,
    }
    measured.runs = [run, {**run, "latency_ms_p50": 210.0}]
    pending = RowResult("graph", {}, "pending: needs Neo4j (NEO4J_* not set)")
    path = tmp_path / "ablation.json"
    save_ablation(path, [measured, pending], True, {"questions": 50, "runs": 2, "labels": "x"})
    table = render_table(path)
    assert "| hybrid | 0.500 | 0.700 | 0.600 | 0.730 | 205 ms ± 5 |" in table
    assert "205 ms ± 5" in table
    assert "| graph | pending |" in table and "needs Neo4j" in table
    assert "provisional" in table
