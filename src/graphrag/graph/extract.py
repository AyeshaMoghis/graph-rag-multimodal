"""Phase 5: LLM triple extraction into Neo4j.

``build_graph`` is the notebook's serial loop: one LLM call per chunk, a fixed
sleep between calls, a checkpoint every 50 chunks, resume from a start index.

``build_graph_async`` replaces it: LLM calls run concurrently under a semaphore,
transient errors retry with exponential backoff and jitter, and a checkpoint of
completed chunk indices makes reruns skip finished work and retry failures.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graphrag.graph.schema import NODE_TYPES, RELATIONS, validate_label, validate_relation
from graphrag.ingest.chunking import Chunk
from graphrag.log import get_logger
from graphrag.reasoning.llm import LLM, AsyncLLM
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


# ── Async extraction (Task 2.4) ─────────────────────────────────────────────


class RetryableError(Exception):
    """Raised by tests or adapters to mark an error as transient."""


def is_retryable(exc: BaseException) -> bool:
    """Rate limits, timeouts and 5xx responses are transient; everything else is not."""
    if isinstance(exc, (RetryableError, TimeoutError, asyncio.TimeoutError)):
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status == 429 or status >= 500):
        return True
    return type(exc).__name__ in {"RateLimitError", "APITimeoutError", "APIConnectionError"}


def retry_after_seconds(exc: BaseException) -> float | None:
    """Honour a server-provided Retry-After header when the SDK exposes one."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    value = headers.get("retry-after") if hasattr(headers, "get") else None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


async def call_with_backoff(
    fn: Callable[[], Awaitable[str]],
    retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: random.Random | None = None,
) -> str:
    """Exponential backoff with jitter on transient errors; re-raise anything else."""
    rng = rng or random.Random()
    for attempt in range(retries + 1):
        try:
            return await fn()
        except Exception as exc:
            if attempt == retries or not is_retryable(exc):
                raise
            delay = retry_after_seconds(exc)
            if delay is None:
                delay = min(max_delay, base_delay * 2**attempt) * (0.5 + rng.random() / 2)
            log.info(
                "llm_retry", attempt=attempt + 1, delay_s=round(delay, 2), error=type(exc).__name__
            )
            await sleep(delay)
    raise AssertionError("unreachable")


@dataclass
class ExtractionCheckpoint:
    """Indices of chunks whose triples are already in the graph."""

    path: Path
    done: set[int] = field(default_factory=set)
    triples: int = 0

    @classmethod
    def load(cls, path: Path) -> ExtractionCheckpoint:
        if not path.exists():
            return cls(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        done = set(data.get("done", []))
        if not done and data.get("chunks_processed"):  # file written by the serial builder
            done = set(range(int(data["chunks_processed"])))
        return cls(path, done, int(data.get("triples_extracted", 0)))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "done": sorted(self.done),
                    "chunks_processed": len(self.done),
                    "triples_extracted": self.triples,
                }
            ),
            encoding="utf-8",
        )
        tmp.replace(self.path)  # atomic, so a crash never leaves a half-written file


async def build_graph_async(
    chunks: Sequence[Chunk],
    llm: AsyncLLM,
    sink: Callable[[Triple, str], None],
    checkpoint_path: Path,
    start_index: int = 0,
    concurrency: int = 4,
    threshold: float = 0.65,
    char_limit: int = 2000,
    checkpoint_every: int = 10,
    retries: int = 5,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> BuildStats:
    """Extract triples concurrently and write them through ``sink`` one at a time.

    LLM calls run under a semaphore; graph writes stay sequential because a
    Neo4j session is not safe to share across tasks. A chunk is marked done
    only after all of its triples were written, so a failed chunk is retried on
    the next run while finished chunks are skipped.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    checkpoint = ExtractionCheckpoint.load(checkpoint_path)
    todo = [i for i in range(start_index, len(chunks)) if i not in checkpoint.done]
    sem = asyncio.Semaphore(concurrency)
    stats = BuildStats()

    async def work(i: int) -> tuple[int, list[Triple] | None]:
        prompt = TRIPLE_PROMPT.replace(
            "REPLACE_TEXT_HERE", _CONTROL_CHARS.sub(" ", chunks[i].text[:char_limit])
        )
        async with sem:
            try:
                raw = await call_with_backoff(
                    lambda: llm.acomplete(prompt), retries=retries, sleep=sleep
                )
            except Exception as exc:
                log.warning("triple_extraction_failed", chunk_id=chunks[i].chunk_id, error=str(exc))
                return i, None
        try:
            return i, filter_triples(parse_triples(raw), threshold)
        except Exception as exc:  # unparseable output is a result, not a transient failure
            log.warning("triple_parse_failed", chunk_id=chunks[i].chunk_id, error=str(exc))
            return i, []

    since_save = 0
    for next_done in asyncio.as_completed([work(i) for i in todo]):
        i, triples = await next_done
        if triples is None:
            stats.errors += 1
            continue
        for t in triples:
            try:
                sink(t, chunks[i].chunk_id)
            except Exception as exc:
                stats.errors += 1
                log.warning("upsert_failed", chunk_id=chunks[i].chunk_id, error=str(exc))
        stats.triples += len(triples)
        stats.chunks_processed += 1
        checkpoint.done.add(i)
        checkpoint.triples += len(triples)
        since_save += 1
        if since_save >= checkpoint_every:
            checkpoint.save()
            since_save = 0
    checkpoint.save()
    return stats
