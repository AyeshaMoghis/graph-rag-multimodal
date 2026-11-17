"""Graph layer: query entity extraction, entity linking, bounded subgraph expansion.

Two modes, selected by ``Settings.graph_mode``:

``legacy``
    Faithful to the notebook, kept for the ablation. Every node name is loaded
    from Neo4j on every query and fuzzy-matched against every query entity.
    Expansion returns the first 30 paths in whatever order Neo4j produces, and
    renders a 2-hop path as ``start -[first relation]-> end``, which hides the
    middle node and ignores edge direction.

``ranked`` (default)
    Entity linking goes lexical (Neo4j full-text index) -> semantic (cached
    node-name embedding matrix) -> fuzzy (only for misses). Expansion orders
    paths by hop count, then by edge support (the smallest ``count`` along the
    path), before truncating, and renders every node and edge direction.

Query entities come from :class:`EntityExtractor`: quoted spans and acronyms
(regex, as before) plus spans from a small NER model. The notebook's rule that
any token longer than four characters is an entity is kept only in
:func:`extract_entities_simple` for the legacy mode.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from rapidfuzz import fuzz

from graphrag.graph.schema import FULLTEXT_INDEX, validate_hops, validate_label
from graphrag.log import get_logger

log = get_logger(__name__)

GraphMode = Literal["legacy", "ranked"]

_STOPWORDS = {"What", "How", "Does", "When", "Why", "This", "That", "With", "From"}
_QUOTED = re.compile(r'"([^"]+)"')
_ACRONYM = re.compile(r"\b([A-Z]{2,})\b")
_CAPITALISED = re.compile(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)")
_LUCENE_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')


# ── Entity extraction ──────────────────────────────────────────────────────


def extract_entities_simple(text: str) -> list[str]:
    """Legacy: quoted spans, capitalised phrases, acronyms, and any token longer than 4 chars."""
    candidates: set[str] = set()
    candidates.update(m.group(1).strip() for m in _QUOTED.finditer(text))
    candidates.update(m.group(1).strip() for m in _CAPITALISED.finditer(text))
    candidates.update(m.group(1) for m in _ACRONYM.finditer(text))
    for tok in text.split():
        tok = tok.strip("?.,!:;")
        if len(tok) > 4 and tok not in _STOPWORDS:
            candidates.add(tok)
    return sorted(e for e in candidates if len(e) > 2)


NER = Callable[[str], list[str]]


class GlinerNER:
    """Zero-shot span extraction with GLiNER, using the graph's own node types as labels."""

    def __init__(
        self,
        model_name: str = "urchade/gliner_small-v2.1",
        labels: Sequence[str] = ("concept", "algorithm", "dataset", "architecture", "platform"),
        threshold: float = 0.4,
    ) -> None:
        self.model_name = model_name
        self.labels = list(labels)
        self.threshold = threshold
        self._model: Any = None

    def __call__(self, text: str) -> list[str]:
        if self._model is None:
            from gliner import GLiNER

            self._model = GLiNER.from_pretrained(self.model_name)
        spans = self._model.predict_entities(text, self.labels, threshold=self.threshold)
        return [str(s["text"]).strip() for s in spans]


class EntityExtractor:
    """Quoted spans and acronyms by regex, plus NER spans. Output is sorted for determinism."""

    def __init__(self, ner: NER | None = None) -> None:
        self.ner = ner

    def __call__(self, text: str) -> list[str]:
        found: set[str] = {m.group(1).strip() for m in _QUOTED.finditer(text)}
        found.update(m.group(1) for m in _ACRONYM.finditer(text))
        if self.ner is not None:
            found.update(self.ner(text))
        return sorted(e for e in found if len(e) > 1 and e not in _STOPWORDS)


# ── Graph store ────────────────────────────────────────────────────────────


class GraphStore(Protocol):
    def named_nodes(self) -> list[dict[str, Any]]: ...

    def fulltext(self, text: str, limit: int) -> list[dict[str, Any]]: ...

    def expand(
        self, name: str, label: str, hops: int, limit: int, ranked: bool = True
    ) -> list[dict[str, Any]]: ...


def escape_lucene(text: str) -> str:
    """Escape Lucene query syntax so user text is matched literally."""
    return _LUCENE_SPECIAL.sub(r"\\\1", text).strip()


_LEGACY_EXPAND = """
MATCH path = (start:{label} {{name: $name}})-[r*1..{hops}]-(related)
WHERE NOT related:Chunk
RETURN start.name AS start_name, type(r[0]) AS rel_type,
       related.name AS end_name, labels(related) AS end_type,
       length(path) AS hops
LIMIT $limit
"""

_RANKED_EXPAND = """
MATCH path = (start:{label} {{name: $name}})-[*1..{hops}]-(related)
WHERE NOT related:Chunk AND related <> start
WITH path, related, relationships(path) AS rs, nodes(path) AS ns
WITH related, rs, ns, length(path) AS hops,
     reduce(m = 2147483647, r IN rs |
            CASE WHEN coalesce(r.count, 1) < m THEN coalesce(r.count, 1) ELSE m END) AS support
RETURN [n IN ns | coalesce(n.name, n.chunk_id)] AS names,
       [r IN rs | type(r)] AS rel_types,
       [i IN range(0, size(rs) - 1) |
            CASE WHEN startNode(rs[i]) = ns[i] THEN '->' ELSE '<-' END] AS dirs,
       related.name AS end_name, labels(related) AS end_type, hops, support
ORDER BY hops ASC, support DESC, end_name ASC
LIMIT $limit
"""


class Neo4jGraphStore:
    def __init__(self, driver: Any) -> None:
        self.driver = driver
        self._nodes: list[dict[str, Any]] | None = None

    def named_nodes(self) -> list[dict[str, Any]]:
        """All named entity nodes. Cached: the legacy mode re-fetched these on every query."""
        if self._nodes is None:
            with self.driver.session() as session:
                self._nodes = list(
                    session.run(
                        "MATCH (n) WHERE n.name IS NOT NULL "
                        "RETURN n.name AS name, labels(n) AS types ORDER BY n.name"
                    ).data()
                )
        return self._nodes

    def fulltext(self, text: str, limit: int) -> list[dict[str, Any]]:
        query = escape_lucene(text)
        if not query:
            return []
        with self.driver.session() as session:
            return list(
                session.run(
                    "CALL db.index.fulltext.queryNodes($index, $q) YIELD node, score "
                    "RETURN node.name AS name, labels(node) AS types, score LIMIT $limit",
                    index=FULLTEXT_INDEX,
                    q=query,
                    limit=limit,
                ).data()
            )

    def expand(
        self, name: str, label: str, hops: int, limit: int, ranked: bool = True
    ) -> list[dict[str, Any]]:
        label, hops = validate_label(label), validate_hops(hops)
        template = _RANKED_EXPAND if ranked else _LEGACY_EXPAND
        with self.driver.session() as session:
            return list(
                session.run(template.format(label=label, hops=hops), name=name, limit=limit).data()
            )


# ── Entity linking ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Link:
    entity: str
    name: str
    label: str
    method: Literal["lexical", "embedding", "fuzzy"]
    score: float


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def link_entities(
    entities: list[str], nodes: list[dict[str, Any]], threshold: int = 70
) -> list[dict[str, Any]]:
    """Legacy: best fuzzy match per entity (token_set_ratio >= threshold), de-duplicated."""
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


class EntityLinker:
    """Lexical (full-text index) -> embedding (cached matrix) -> fuzzy fallback."""

    def __init__(
        self,
        store: GraphStore,
        encoder: Any | None = None,
        lexical_threshold: int = 90,
        embedding_threshold: float = 0.80,
        fuzzy_threshold: int = 70,
        cache_dir: Path | None = None,
    ) -> None:
        self.store = store
        self.encoder = encoder
        self.lexical_threshold = lexical_threshold
        self.embedding_threshold = embedding_threshold
        self.fuzzy_threshold = fuzzy_threshold
        self.cache_dir = cache_dir
        self._matrix: NDArray[np.float32] | None = None

    def _node_matrix(self) -> NDArray[np.float32]:
        if self._matrix is None:
            assert self.encoder is not None
            names = [str(n["name"]) for n in self.store.named_nodes()]
            key = hashlib.sha1("\n".join(names).encode("utf-8")).hexdigest()[:16]
            path = self.cache_dir / f"node_embeddings_{key}.npy" if self.cache_dir else None
            if path is not None and path.exists():
                self._matrix = np.load(path)
            else:
                self._matrix = np.asarray(self.encoder.encode(names), dtype=np.float32)
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, self._matrix)
        return self._matrix

    def link_one(self, entity: str) -> Link | None:
        # 1. Lexical: full-text candidates whose normalised name nearly equals the entity.
        for cand in self.store.fulltext(entity, 5):
            score = fuzz.ratio(_norm(entity), _norm(str(cand["name"])))
            if score >= self.lexical_threshold:
                return Link(entity, cand["name"], cand["types"][0], "lexical", float(score))

        nodes = self.store.named_nodes()
        if not nodes:
            return None

        # 2. Semantic: cosine similarity against the cached node-name embeddings.
        if self.encoder is not None:
            matrix = self._node_matrix()
            query = np.asarray(self.encoder.encode([entity]), dtype=np.float32)[0]
            sims = matrix @ query
            best = int(np.argmax(sims))
            if float(sims[best]) >= self.embedding_threshold:
                n = nodes[best]
                return Link(entity, n["name"], n["types"][0], "embedding", float(sims[best]))

        # 3. Fuzzy fallback for misses only (the original rule).
        best_score, best_node = 0.0, None
        for n in nodes:
            score = fuzz.token_set_ratio(entity.lower(), str(n["name"]).lower())
            if score > best_score:
                best_score, best_node = score, n
        if best_node is not None and best_score >= self.fuzzy_threshold:
            return Link(entity, best_node["name"], best_node["types"][0], "fuzzy", best_score)
        return None

    def link(self, entities: Sequence[str], top_n: int) -> list[Link]:
        links: list[Link] = []
        seen: set[str] = set()
        for entity in entities:
            hit = self.link_one(entity)
            if hit is not None and hit.name not in seen:
                seen.add(hit.name)
                links.append(hit)
        return links[:top_n]


# ── Retrieval ──────────────────────────────────────────────────────────────


@dataclass
class GraphResult:
    paths: list[str] = field(default_factory=list)
    nodes: list[str] = field(default_factory=list)
    raw: list[dict[str, Any]] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)


def render_path(names: Sequence[str], rel_types: Sequence[str], dirs: Sequence[str]) -> str:
    """``A -[USES]-> B <-[IS_A]- C``: every node and the true edge direction."""
    out = str(names[0])
    for rel, direction, node in zip(rel_types, dirs, names[1:], strict=True):
        out += f" -[{rel}]-> {node}" if direction == "->" else f" <-[{rel}]- {node}"
    return out


def graph_retrieve(
    entities: list[str],
    store: GraphStore,
    hops: int = 2,
    top_n_entities: int = 5,
    fuzzy_threshold: int = 70,
    path_limit: int = 30,
    mode: GraphMode = "ranked",
    linker: EntityLinker | None = None,
) -> GraphResult:
    if not entities:
        return GraphResult()
    if mode == "legacy":
        return _legacy_retrieve(entities, store, hops, top_n_entities, fuzzy_threshold, path_limit)

    linker = linker or EntityLinker(store, fuzzy_threshold=fuzzy_threshold)
    links = linker.link(entities, top_n_entities)
    raw: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    visited: list[str] = []
    for link in links:
        visited.append(link.name)
        for row in store.expand(link.name, link.label, hops, path_limit, ranked=True):
            if not row.get("end_name"):
                continue
            path = render_path(row["names"], row["rel_types"], row["dirs"])
            if path in seen_paths:
                continue
            seen_paths.add(path)
            raw.append({**row, "path": path, "entity": link.name})
            visited.extend(str(n) for n in row["names"] if n)
    raw.sort(key=lambda p: (p["hops"], -p["support"]))  # stable across entities
    return GraphResult(
        paths=[p["path"] for p in raw],
        nodes=sorted(set(visited)),
        raw=raw,
        matched_entities=[link.name for link in links],
        links=links,
    )


def _legacy_retrieve(
    entities: list[str],
    store: GraphStore,
    hops: int,
    top_n_entities: int,
    fuzzy_threshold: int,
    path_limit: int,
) -> GraphResult:
    matched = link_entities(entities, store.named_nodes(), fuzzy_threshold)[:top_n_entities]
    raw: list[dict[str, Any]] = []
    visited: list[str] = []
    for node in matched:
        visited.append(node["name"])
        for row in store.expand(node["name"], node["types"][0], hops, path_limit, ranked=False):
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
