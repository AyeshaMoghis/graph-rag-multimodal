"""Graph layer: query entity extraction, fuzzy entity linking, subgraph expansion.

Faithful port of the notebook. Known scalability issues (all nodes loaded per
query, token-length entity heuristic, unranked LIMIT 30) are fixed in Task 2.4.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from rapidfuzz import fuzz

from graphrag.graph.schema import validate_hops, validate_label

_STOPWORDS = {"What", "How", "Does", "When", "Why", "This", "That", "With", "From"}


def extract_entities_simple(text: str) -> list[str]:
    """Quoted spans, capitalised phrases, acronyms, and any token longer than 4 chars."""
    candidates: set[str] = set()
    candidates.update(m.group(1).strip() for m in re.finditer(r'"([^"]+)"', text))
    candidates.update(
        m.group(1).strip() for m in re.finditer(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text)
    )
    candidates.update(m.group(1) for m in re.finditer(r"\b([A-Z]{2,})\b", text))
    for tok in text.split():
        tok = tok.strip("?.,!:;")
        if len(tok) > 4 and tok not in _STOPWORDS:
            candidates.add(tok)
    return sorted(e for e in candidates if len(e) > 2)


class GraphStore(Protocol):
    def named_nodes(self) -> list[dict[str, Any]]: ...

    def expand(self, name: str, label: str, hops: int, limit: int) -> list[dict[str, Any]]: ...


class Neo4jGraphStore:
    def __init__(self, driver: Any) -> None:
        self.driver = driver

    def named_nodes(self) -> list[dict[str, Any]]:
        with self.driver.session() as session:
            return list(
                session.run(
                    "MATCH (n) WHERE n.name IS NOT NULL RETURN n.name AS name, labels(n) AS types"
                ).data()
            )

    def expand(self, name: str, label: str, hops: int, limit: int) -> list[dict[str, Any]]:
        label, hops = validate_label(label), validate_hops(hops)
        with self.driver.session() as session:
            return list(
                session.run(
                    f"""
                    MATCH path = (start:{label} {{name: $name}})-[r*1..{hops}]-(related)
                    WHERE NOT related:Chunk
                    RETURN start.name AS start_name, type(r[0]) AS rel_type,
                           related.name AS end_name, labels(related) AS end_type,
                           length(path) AS hops
                    LIMIT $limit
                    """,
                    name=name,
                    limit=limit,
                ).data()
            )


@dataclass
class GraphResult:
    paths: list[str] = field(default_factory=list)
    nodes: list[str] = field(default_factory=list)
    raw: list[dict[str, Any]] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)


def link_entities(
    entities: list[str], nodes: list[dict[str, Any]], threshold: int = 70
) -> list[dict[str, Any]]:
    """Best fuzzy match per entity (token_set_ratio >= threshold), de-duplicated by name."""
    matched: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entity in entities:
        best_score, best_node = 0.0, None
        for row in nodes:
            score = fuzz.token_set_ratio(entity.lower(), str(row["name"]).lower())
            if score > best_score:
                best_score, best_node = score, row
        if best_node is not None and best_score >= threshold and best_node["name"] not in seen:
            seen.add(best_node["name"])
            matched.append(best_node)
    return matched


def graph_retrieve(
    entities: list[str],
    store: GraphStore,
    hops: int = 2,
    top_n_entities: int = 5,
    fuzzy_threshold: int = 70,
    path_limit: int = 30,
) -> GraphResult:
    if not entities:
        return GraphResult()
    matched = link_entities(entities, store.named_nodes(), fuzzy_threshold)[:top_n_entities]
    if not matched:
        return GraphResult()

    raw: list[dict[str, Any]] = []
    visited: list[str] = []
    for node in matched:
        label = node["types"][0]
        visited.append(node["name"])
        for row in store.expand(node["name"], label, hops, path_limit):
            if row.get("end_name"):
                raw.append(
                    {
                        "path": f"{row['start_name']} -[{row['rel_type']}]-> {row['end_name']}",
                        "hops": row["hops"],
                        "end_name": row["end_name"],
                        "end_type": row["end_type"],
                    }
                )
                visited.append(row["end_name"])

    raw.sort(key=lambda p: p["hops"])
    return GraphResult(
        paths=[p["path"] for p in raw],
        nodes=sorted(set(visited)),
        raw=raw,
        matched_entities=[n["name"] for n in matched],
    )
