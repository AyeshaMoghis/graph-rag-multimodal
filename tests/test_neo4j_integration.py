"""Integration test against a real Neo4j (slow). CI starts Neo4j as a service container.

Skipped unless NEO4J_TEST_URI, NEO4J_TEST_USER and NEO4J_TEST_PASSWORD are set.
The test wipes the database it connects to, so never point it at real data.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest

from graphrag.graph.extract import Triple, upsert_triple
from graphrag.graph.retrieve import EntityLinker, Neo4jGraphStore, graph_retrieve
from graphrag.graph.schema import apply_schema, graph_stats

pytestmark = pytest.mark.slow

URI = os.environ.get("NEO4J_TEST_URI")
USER = os.environ.get("NEO4J_TEST_USER")
PASSWORD = os.environ.get("NEO4J_TEST_PASSWORD")


@pytest.fixture(scope="module")
def driver() -> Iterator[Any]:
    if not (URI and USER and PASSWORD):
        pytest.skip("NEO4J_TEST_* not set")
    from neo4j import GraphDatabase

    drv = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    with drv.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    apply_schema(drv)
    with drv.session() as session:
        for t, chunk in [
            (Triple("MBTI", "Concept", "IS_A", "Personality Model", "Concept", 0.9), "c1"),
            (
                Triple("Personality Model", "Concept", "PART_OF", "User Modeling", "Concept", 0.8),
                "c2",
            ),
            (Triple("MBTI", "Concept", "IS_A", "Personality Model", "Concept", 0.9), "c3"),
        ]:
            upsert_triple(session, t, chunk)
    yield drv
    drv.close()


def test_schema_and_upserts(driver: Any) -> None:
    stats = graph_stats(driver)
    assert stats["labels"]["Concept"] == 3
    assert stats["labels"]["Chunk"] == 3
    with driver.session() as session:
        count = session.run(
            "MATCH (:Concept {name: 'MBTI'})-[r:IS_A]->() RETURN r.count AS c"
        ).single()["c"]
    assert count == 2  # second identical triple increments the edge count


def test_fulltext_lexical_link_and_ranked_expansion(driver: Any) -> None:
    store = Neo4jGraphStore(driver)
    assert store.fulltext("personality", 5)[0]["name"] == "Personality Model"
    link = EntityLinker(store).link_one("mbti")
    assert link is not None and (link.name, link.method) == ("MBTI", "lexical")

    result = graph_retrieve(["MBTI"], store, hops=2, path_limit=30)
    assert result.paths[0] == "MBTI -[IS_A]-> Personality Model"
    assert "MBTI -[IS_A]-> Personality Model -[PART_OF]-> User Modeling" in result.paths


def test_cypher_injection_in_entity_text_is_inert(driver: Any) -> None:
    store = Neo4jGraphStore(driver)
    store.fulltext("x') DETACH DELETE n //", 5)  # escaped and passed as a parameter
    assert graph_stats(driver)["nodes"] > 0
