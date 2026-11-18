from __future__ import annotations

from pathlib import Path

import pytest

from graphrag.config import Settings
from graphrag.eval.harness import summarize
from graphrag.eval.metrics import context_keyword_recall, keyword_recall
from graphrag.index.captions import is_useless, ocr_passes_threshold
from graphrag.reasoning.answer import parse_answer


def test_keyword_recall_matches_original_definition() -> None:
    assert keyword_recall("MBTI has sixteen types", ["mbti", "types", "Jung"]) == 0.667
    assert keyword_recall("anything", []) == 1.0


def test_context_keyword_recall_joins_chunks() -> None:
    assert context_keyword_recall(["about MBTI", "about Jung"], ["MBTI", "Jung"]) == 1.0


def test_summarize_retrieval_mode() -> None:
    rows = [
        {
            "id": "q1",
            "question_type": "factual",
            "context_keyword_recall": 1.0,
            "latency_ms": 10.0,
            "duplicate_slots": 1,
            "both_signal_hits": 0,
        },
        {
            "id": "q2",
            "question_type": "multi-hop",
            "context_keyword_recall": 0.5,
            "latency_ms": 30.0,
            "duplicate_slots": 0,
            "both_signal_hits": 2,
        },
        {"id": "q3", "question_type": "factual", "error": "boom"},
    ]
    s = summarize(rows, "retrieval")
    assert s["context_keyword_recall"] == 0.75
    assert s["errors"] == 1 and s["latency_ms_p50"] == 20.0
    assert s["by_type"]["multi-hop"]["n"] == 1
    assert s["duplicate_slots_total"] == 1


def test_parse_answer_falls_back_on_bad_json() -> None:
    out = parse_answer("plain text")
    assert out["answer"] == "plain text" and out["confidence"] == "low"
    assert parse_answer('```json\n{"answer": "x", "confidence": "high"}\n```')["answer"] == "x"


def test_caption_filters() -> None:
    assert is_useless("diagram")
    assert is_useless("a b c")
    assert is_useless("12 34 56 78 !!")
    assert not is_useless("a bar chart of user credibility types")
    assert ocr_passes_threshold("word " * 15)
    assert not ocr_passes_threshold("1 2 3 | -- ok " * 5)


def test_settings_read_credentials_from_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)  # no stray .env
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "pw")
    monkeypatch.setenv("GRAPHRAG_FUSION", "weighted")
    s = Settings()
    assert s.groq_api_key is not None and s.groq_api_key.get_secret_value() == "test-key"
    assert "test-key" not in repr(s)
    assert s.graph_enabled and s.fusion == "weighted"


def test_settings_defaults_have_no_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    for var in ("GROQ_API_KEY", "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    s = Settings()
    assert s.groq_api_key is None and not s.graph_enabled
    assert s.fusion == "rrf" and s.rrf_k == 60


def test_repetitive_captions_are_useless() -> None:
    from graphrag.index.captions import is_repetitive

    assert is_useless("gender gender gender gender gender gender gender gender")
    assert is_repetitive("global market market market market market market")
    assert not is_repetitive("a bar chart comparing user credibility types across platforms")
