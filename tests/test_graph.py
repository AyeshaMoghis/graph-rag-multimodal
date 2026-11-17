"""Graph extraction parsing and graph retrieval, with Neo4j and the LLM mocked."""

from __future__ import annotations

from typing import Any

import pytest

from graphrag.graph.extract import (
    Triple,
    extract_triples,
    filter_triples,
    parse_triples,
    upsert_triple,
)
from graphrag.graph.retrieve import extract_entities_simple, graph_retrieve, link_entities
from graphrag.graph.schema import validate_hops, validate_label, validate_relation


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


GOOD = {
    "subject": "Adam",
    "subject_type": "Algorithm",
    "relation": "USES",
    "object": "gradient descent",
    "object_type": "Concept",
    "confidence": 0.9,
}


def test_parse_triples_strips_fences() -> None:
    assert parse_triples('```json\n[{"a": 1}]\n```') == [{"a": 1}]


def test_filter_enforces_threshold_and_schema() -> None:
    rows = [
        GOOD,
        {**GOOD, "confidence": 0.64},
        {**GOOD, "relation": "LOVES"},
        {**GOOD, "subject_type": "Person"},
        {**GOOD, "object": ""},
        {**GOOD, "confidence": "high"},
    ]
    kept = filter_triples(rows, threshold=0.65)
    assert len(kept) == 1 and kept[0].subject == "Adam"


def test_extract_triples_survives_bad_json() -> None:
    assert extract_triples("text", FakeLLM("not json")) == []


def test_extract_triples_truncates_input() -> None:
    llm = FakeLLM("[]")
    extract_triples("~" * 5000, llm, char_limit=2000)
    assert llm.prompts[0].count("~") == 2000


def test_validators_reject_injection() -> None:
    assert validate_label("Concept") == "Concept"
    with pytest.raises(ValueError):
        validate_label("Concept) DETACH DELETE (n")
    with pytest.raises(ValueError):
        validate_relation("USES]->() DELETE x//")
    with pytest.raises(ValueError):
        validate_hops(5)


class RecordingSession:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def run(self, query: str, **params: Any) -> None:
        self.queries.append(query)


def test_upsert_skips_relation_merge_for_described_in() -> None:
    session = RecordingSession()
    upsert_triple(session, Triple("A", "Concept", "DESCRIBED_IN", "B", "Concept", 0.9), "c1")
    assert not any("MERGE (s)-[r:" in q for q in session.queries)
    session = RecordingSession()
    upsert_triple(session, Triple("A", "Concept", "USES", "B", "Concept", 0.9), "c1")
    assert any("MERGE (s)-[r:USES]->(o)" in q for q in session.queries)


def test_entity_extraction_finds_acronyms_quotes_and_long_tokens() -> None:
    ents = extract_entities_simple('How does "echo chamber" relate to MBTI personality?')
    assert {"echo chamber", "MBTI", "personality"} <= set(ents)
    # Faithful to the original: the stopword list only filters the long-token rule,
    # so the capitalised-phrase regex still emits question words. Task 2.4 replaces this.
    assert "How" in ents
    assert ents == sorted(ents)  # deterministic order


class FakeGraphStore:
    def __init__(self) -> None:
        self.expanded: list[tuple[str, str, int, int]] = []

    def named_nodes(self) -> list[dict[str, Any]]:
        return [
            {"name": "MBTI", "types": ["Concept"]},
            {"name": "Personality Types", "types": ["Concept"]},
            {"name": "BM25", "types": ["Algorithm"]},
        ]

    def expand(self, name: str, label: str, hops: int, limit: int) -> list[dict[str, Any]]:
        self.expanded.append((name, label, hops, limit))
        if name == "MBTI":
            return [
                {
                    "start_name": "MBTI",
                    "rel_type": "IS_A",
                    "end_name": "Framework",
                    "end_type": ["Concept"],
                    "hops": 2,
                },
                {
                    "start_name": "MBTI",
                    "rel_type": "USES",
                    "end_name": "Personality Types",
                    "end_type": ["Concept"],
                    "hops": 1,
                },
            ]
        return []


def test_link_entities_threshold_and_dedup() -> None:
    nodes = FakeGraphStore().named_nodes()
    matched = link_entities(["mbti", "MBTI", "zzzz"], nodes, threshold=70)
    assert [m["name"] for m in matched] == ["MBTI"]


def test_graph_retrieve_orders_by_hops() -> None:
    store = FakeGraphStore()
    result = graph_retrieve(["MBTI"], store, hops=2, path_limit=30)
    assert result.paths[0] == "MBTI -[USES]-> Personality Types"
    assert result.matched_entities == ["MBTI"]
    assert store.expanded[0] == ("MBTI", "Concept", 2, 30)


def test_graph_retrieve_no_entities_or_matches() -> None:
    assert graph_retrieve([], FakeGraphStore()).paths == []
    assert graph_retrieve(["qqqqqq"], FakeGraphStore()).paths == []
