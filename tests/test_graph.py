"""Graph extraction parsing and graph retrieval, with Neo4j and the LLM mocked."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from graphrag.graph.extract import (
    Triple,
    extract_triples,
    filter_triples,
    parse_triples,
    upsert_triple,
)
from graphrag.graph.retrieve import (
    EntityExtractor,
    EntityLinker,
    Neo4jGraphStore,
    escape_lucene,
    extract_entities_simple,
    graph_retrieve,
    link_entities,
    render_path,
)
from graphrag.graph.schema import (
    FULLTEXT_INDEX_DDL,
    validate_hops,
    validate_label,
    validate_relation,
)


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


# ── Triple extraction ───────────────────────────────────────────────────────


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


def test_fulltext_index_ddl_covers_every_entity_label() -> None:
    assert "CREATE FULLTEXT INDEX entity_names IF NOT EXISTS" in FULLTEXT_INDEX_DDL
    assert "Algorithm|Architecture|Concept|Dataset" in FULLTEXT_INDEX_DDL


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


# ── Entity extraction ───────────────────────────────────────────────────────


def test_legacy_entity_extraction_keeps_original_quirks() -> None:
    ents = extract_entities_simple('How does "echo chamber" relate to MBTI personality?')
    assert {"echo chamber", "MBTI", "personality"} <= set(ents)
    # The capitalised-phrase regex emits question words; the long-token rule adds "relate".
    assert "How" in ents and "relate" in ents
    assert ents == sorted(ents)


def test_entity_extractor_uses_ner_and_regexes() -> None:
    extractor = EntityExtractor(ner=lambda text: ["user credibility", "disinformation"])
    ents = extractor('How does "echo chamber" relate to MBTI?')
    assert ents == sorted(["MBTI", "disinformation", "echo chamber", "user credibility"])


# ── Fakes for the graph store and encoder ───────────────────────────────────


class FakeGraphStore:
    """In-memory stand-in for Neo4j. Rows mimic both expansion query shapes."""

    def __init__(self) -> None:
        self.expanded: list[tuple[str, str, int, int, bool]] = []

    def named_nodes(self) -> list[dict[str, Any]]:
        return [
            {"name": "BM25", "types": ["Algorithm"]},
            {"name": "Cyberbullying", "types": ["Concept"]},
            {"name": "MBTI", "types": ["Concept"]},
            {"name": "Personality Types", "types": ["Concept"]},
        ]

    def fulltext(self, text: str, limit: int) -> list[dict[str, Any]]:
        words = set(text.lower().replace("-", " ").split()) | {text.lower().replace("-", "")}
        return [
            {**n, "score": 1.0}
            for n in self.named_nodes()
            if words & set(n["name"].lower().split())
        ][:limit]

    def expand(
        self, name: str, label: str, hops: int, limit: int, ranked: bool = True
    ) -> list[dict[str, Any]]:
        self.expanded.append((name, label, hops, limit, ranked))
        if name != "MBTI":
            return []
        if not ranked:
            return [
                {
                    "start_name": "MBTI",
                    "rel_type": "USES",
                    "end_name": "Personality Types",
                    "end_type": ["Concept"],
                    "hops": 1,
                },
                {
                    "start_name": "MBTI",
                    "rel_type": "USES",
                    "end_name": "Framework",
                    "end_type": ["Concept"],
                    "hops": 2,
                },
            ]
        return [
            {
                "names": ["MBTI", "Personality Types"],
                "rel_types": ["USES"],
                "dirs": ["->"],
                "end_name": "Personality Types",
                "end_type": ["Concept"],
                "hops": 1,
                "support": 3,
            },
            {
                "names": ["MBTI", "Personality Types", "Framework"],
                "rel_types": ["USES", "IS_A"],
                "dirs": ["->", "<-"],
                "end_name": "Framework",
                "end_type": ["Concept"],
                "hops": 2,
                "support": 1,
            },
        ]


class FakeEncoder:
    """Maps a few strings to fixed unit vectors so cosine similarity is predictable."""

    VECS: ClassVar[dict[str, list[float]]] = {
        "mbti": [1.0, 0.0, 0.0],
        "myers-briggs type indicator": [0.95, 0.312, 0.0],
        "personality types": [0.0, 1.0, 0.0],
        "bm25": [0.0, 0.0, 1.0],
        "cyberbullying": [0.6, 0.0, 0.8],
    }

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        rows = [self.VECS.get(t.lower(), [0.577, 0.577, 0.577]) for t in texts]
        out = np.array(rows, dtype=np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


# ── Legacy mode: original behaviour pinned for the ablation ─────────────────


def test_legacy_link_entities_threshold_and_dedup() -> None:
    nodes = FakeGraphStore().named_nodes()
    matched = link_entities(["mbti", "MBTI", "zzzz"], nodes, threshold=70)
    assert [m["name"] for m in matched] == ["MBTI"]


def test_legacy_retrieve_renders_first_relation_only() -> None:
    store = FakeGraphStore()
    result = graph_retrieve(["MBTI"], store, hops=2, path_limit=30, mode="legacy")
    # The 2-hop path hides the middle node: this is the original rendering bug.
    assert result.paths == ["MBTI -[USES]-> Personality Types", "MBTI -[USES]-> Framework"]
    assert store.expanded[0] == ("MBTI", "Concept", 2, 30, False)


# ── Ranked mode ─────────────────────────────────────────────────────────────


def test_ranked_retrieve_renders_full_paths_with_direction() -> None:
    store = FakeGraphStore()
    result = graph_retrieve(["MBTI"], store, hops=2, path_limit=30)
    assert result.paths == [
        "MBTI -[USES]-> Personality Types",
        "MBTI -[USES]-> Personality Types <-[IS_A]- Framework",
    ]
    assert result.links[0].method == "lexical"
    assert store.expanded[0][-1] is True


def test_render_path() -> None:
    assert render_path(["A", "B", "C"], ["USES", "IS_A"], ["->", "<-"]) == (
        "A -[USES]-> B <-[IS_A]- C"
    )


def test_linker_lexical_match_uses_fulltext() -> None:
    link = EntityLinker(FakeGraphStore()).link_one("cyber-bullying")
    assert link is not None
    assert (link.name, link.method) == ("Cyberbullying", "lexical")


def test_linker_embedding_match_for_synonyms() -> None:
    linker = EntityLinker(FakeGraphStore(), encoder=FakeEncoder(), embedding_threshold=0.9)
    link = linker.link_one("Myers-Briggs Type Indicator")
    assert link is not None
    assert (link.name, link.method) == ("MBTI", "embedding")


def test_linker_falls_back_to_fuzzy_only_for_misses() -> None:
    linker = EntityLinker(FakeGraphStore(), encoder=FakeEncoder(), embedding_threshold=0.99)
    link = linker.link_one("personaltypes")
    assert link is not None
    assert (link.name, link.method) == ("Personality Types", "fuzzy")
    assert linker.link_one("qqqqqq") is None


def test_linker_caches_node_matrix(tmp_path: Path) -> None:
    enc = FakeEncoder()
    linker = EntityLinker(
        FakeGraphStore(), encoder=enc, embedding_threshold=0.99, cache_dir=tmp_path
    )
    linker.link_one("zzz one")
    linker.link_one("zzz two")
    assert enc.calls == 3  # one matrix build, then one encode per query entity
    assert list(tmp_path.glob("node_embeddings_*.npy"))

    fresh_enc = FakeEncoder()
    fresh = EntityLinker(
        FakeGraphStore(), encoder=fresh_enc, embedding_threshold=0.99, cache_dir=tmp_path
    )
    fresh.link_one("zzz")
    assert fresh_enc.calls == 1  # matrix loaded from disk


def test_escape_lucene() -> None:
    assert escape_lucene('a:b (c) "d"') == r"a\:b \(c\) \"d\""


def test_graph_retrieve_no_entities_or_matches() -> None:
    assert graph_retrieve([], FakeGraphStore()).paths == []
    assert graph_retrieve(["qqqqqq"], FakeGraphStore()).paths == []
    assert graph_retrieve(["qqqqqq"], FakeGraphStore(), mode="legacy").paths == []


# ── Neo4j store against a fake driver ───────────────────────────────────────


class FakeResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def data(self) -> list[dict[str, Any]]:
        return self.rows


class FakeSession:
    def __init__(self, log: list[tuple[str, dict[str, Any]]]) -> None:
        self.log = log

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def run(self, query: str, **params: Any) -> FakeResult:
        self.log.append((query, params))
        return FakeResult([{"name": "MBTI", "types": ["Concept"]}])


class FakeDriver:
    def __init__(self) -> None:
        self.log: list[tuple[str, dict[str, Any]]] = []

    def session(self) -> FakeSession:
        return FakeSession(self.log)


def test_neo4j_store_caches_nodes_and_ranks_paths() -> None:
    driver = FakeDriver()
    store = Neo4jGraphStore(driver)
    store.named_nodes()
    store.named_nodes()
    assert len(driver.log) == 1  # the legacy code re-fetched every node per query

    store.expand("MBTI", "Concept", 2, 30)
    query, params = driver.log[-1]
    assert "ORDER BY hops ASC, support DESC" in query
    assert params == {"name": "MBTI", "limit": 30}

    store.fulltext('MBTI "types"', 5)
    assert driver.log[-1][1]["q"] == r"MBTI \"types\""

    with pytest.raises(ValueError):
        store.expand("MBTI", "Concept) DETACH DELETE (x", 2, 30)
