"""Knowledge-graph schema: allowed node labels, relation types and constraints.

Labels and relation types cannot be Cypher parameters, so any code that puts
one into a query string must pass it through :func:`validate_label` or
:func:`validate_relation` first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graphrag.config import Settings

NODE_TYPES: frozenset[str] = frozenset({"Concept", "Algorithm", "Dataset", "Architecture"})
RELATIONS: frozenset[str] = frozenset({"USES", "IS_A", "CONTRASTS_WITH", "PART_OF", "DESCRIBED_IN"})
CHUNK_LABEL = "Chunk"

CONSTRAINTS: tuple[str, ...] = (
    *(
        f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.name IS UNIQUE"
        for label in sorted(NODE_TYPES)
    ),
    "CREATE CONSTRAINT IF NOT EXISTS FOR (ch:Chunk) REQUIRE ch.chunk_id IS UNIQUE",
)

# Full-text index over entity names, used for lexical entity linking.
# Neo4j 5 syntax; db.index.fulltext.createNodeIndex was removed in Neo4j 5.
FULLTEXT_INDEX = "entity_names"
FULLTEXT_INDEX_DDL = (
    f"CREATE FULLTEXT INDEX {FULLTEXT_INDEX} IF NOT EXISTS "
    f"FOR (n:{'|'.join(sorted(NODE_TYPES))}) ON EACH [n.name]"
)


def validate_label(label: str) -> str:
    if label not in NODE_TYPES:
        raise ValueError(f"node label not allowed: {label!r}")
    return label


def validate_relation(relation: str) -> str:
    if relation not in RELATIONS:
        raise ValueError(f"relation type not allowed: {relation!r}")
    return relation


def validate_hops(hops: int) -> int:
    if not isinstance(hops, int) or not 1 <= hops <= 3:
        raise ValueError(f"hops must be an int in [1, 3], got {hops!r}")
    return hops


def apply_schema(driver: Any) -> None:
    with driver.session() as session:
        for cypher in (*CONSTRAINTS, FULLTEXT_INDEX_DDL):
            session.run(cypher)


def graph_stats(driver: Any) -> dict[str, Any]:
    with driver.session() as session:
        nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        labels = session.run(
            "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS c ORDER BY c DESC"
        ).data()
    return {"nodes": nodes, "relationships": rels, "labels": {r["label"]: r["c"] for r in labels}}


def connect(settings: Settings) -> Any:
    """Open a Neo4j driver from settings, failing clearly if credentials are missing."""
    from neo4j import GraphDatabase

    if not (settings.neo4j_uri and settings.neo4j_user and settings.neo4j_password):
        raise RuntimeError("NEO4J_URI, NEO4J_USER and NEO4J_PASSWORD must all be set")
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
    )
