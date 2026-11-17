"""Phase 5: LLM triple extraction into Neo4j.

Ported from the notebook: one LLM call per chunk, a fixed sleep between calls,
a checkpoint file every 50 chunks, and resume from a start index. Task 2.4
makes this async with bounded concurrency and backoff.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graphrag.graph.schema import NODE_TYPES, RELATIONS, validate_label, validate_relation
from graphrag.ingest.chunking import Chunk
from graphrag.log import get_logger
from graphrag.reasoning.llm import LLM
from graphrag.reasoning.prompts import TRIPLE_PROMPT

log = get_logger(__name__)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class Triple:
    subject: str
    subject_type: str
    relation: str
    object: str
    object_type: str
    confidence: float


def strip_fences(raw: str) -> str:
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    return re.sub(r"\n?```$", "", raw).strip()


def parse_triples(raw: str) -> list[dict[str, Any]]:
    data = json.loads(_CONTROL_CHARS.sub(" ", strip_fences(raw)))
    return data if isinstance(data, list) else []


def filter_triples(candidates: Sequence[dict[str, Any]], threshold: float = 0.65) -> list[Triple]:
    """Keep triples that meet the confidence threshold and the constrained schema."""
    kept: list[Triple] = []
    for t in candidates:
        try:
            confidence = float(t.get("confidence", 0))
        except (TypeError, ValueError):
            continue
        if (
            confidence >= threshold
            and t.get("relation") in RELATIONS
            and t.get("subject_type") in NODE_TYPES
            and t.get("object_type") in NODE_TYPES
            and t.get("subject")
            and t.get("object")
        ):
            kept.append(
                Triple(
                    str(t["subject"]),
                    t["subject_type"],
                    t["relation"],
                    str(t["object"]),
                    t["object_type"],
                    confidence,
                )
            )
    return kept


def extract_triples(
    text: str, llm: LLM, threshold: float = 0.65, char_limit: int = 2000
) -> list[Triple]:
    prompt = TRIPLE_PROMPT.replace("REPLACE_TEXT_HERE", _CONTROL_CHARS.sub(" ", text[:char_limit]))
    try:
        return filter_triples(parse_triples(llm.complete(prompt)), threshold)
    except Exception as exc:
        log.warning("triple_extraction_failed", error=str(exc))
        return []


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", name.strip().lower()).strip()


def upsert_triple(session: Any, t: Triple, chunk_id: str) -> None:
    s_label, o_label = validate_label(t.subject_type), validate_label(t.object_type)
    relation = validate_relation(t.relation)
    for name, label in ((t.subject, s_label), (t.object, o_label)):
        session.run(
            f"MERGE (n:{label} {{name: $name}}) ON CREATE SET n.normalized = $norm",
            name=name,
            norm=normalize_name(name),
        )
    if relation != "DESCRIBED_IN":
        session.run(
            f"""
            MATCH (s:{s_label} {{name: $subj}})
            MATCH (o:{o_label} {{name: $obj}})
            MERGE (s)-[r:{relation}]->(o)
            ON CREATE SET r.chunk_id = $chunk_id, r.count = 1
            ON MATCH SET r.count = r.count + 1
            """,
            subj=t.subject,
            obj=t.object,
            chunk_id=chunk_id,
        )
    session.run(
        f"""
        MATCH (n:{s_label} {{name: $name}})
        MERGE (ch:Chunk {{chunk_id: $cid}})
        MERGE (n)-[:DESCRIBED_IN]->(ch)
        """,
        name=t.subject,
        cid=chunk_id,
    )


@dataclass
class BuildStats:
    chunks_processed: int = 0
    triples: int = 0
    errors: int = 0


def build_graph(
    chunks: Sequence[Chunk],
    llm: LLM,
    driver: Any,
    checkpoint_path: Path,
    start_index: int = 0,
    threshold: float = 0.65,
    sleep_seconds: float = 1.0,
    checkpoint_every: int = 50,
) -> BuildStats:
    stats = BuildStats()
    with driver.session() as session:
        for offset, chunk in enumerate(chunks[start_index:]):
            index = start_index + offset
            triples = extract_triples(chunk.text, llm, threshold)
            stats.triples += len(triples)
            for t in triples:
                try:
                    upsert_triple(session, t, chunk.chunk_id)
                except Exception as exc:
                    stats.errors += 1
                    log.warning("upsert_failed", chunk_id=chunk.chunk_id, error=str(exc))
            stats.chunks_processed += 1
            if sleep_seconds:
                time.sleep(sleep_seconds)
            if (offset + 1) % checkpoint_every == 0:
                _checkpoint(checkpoint_path, index + 1, stats)
    _checkpoint(checkpoint_path, start_index + stats.chunks_processed, stats)
    return stats


def _checkpoint(path: Path, next_index: int, stats: BuildStats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"chunks_processed": next_index, "triples_extracted": stats.triples}),
        encoding="utf-8",
    )
    log.info("graph_checkpoint", chunks_processed=next_index, triples=stats.triples)
