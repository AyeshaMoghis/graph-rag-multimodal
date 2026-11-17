"""Async triple extraction: concurrency bound, backoff, and resumable checkpoints."""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any, ClassVar

import pytest

from graphrag.graph.extract import (
    ExtractionCheckpoint,
    RetryableError,
    Triple,
    build_graph_async,
    call_with_backoff,
    is_retryable,
)
from graphrag.ingest.chunking import Chunk

TRIPLE_JSON = json.dumps(
    [
        {
            "subject": "MBTI",
            "subject_type": "Concept",
            "relation": "IS_A",
            "object": "Personality Model",
            "object_type": "Concept",
            "confidence": 0.9,
        }
    ]
)


def chunks(n: int) -> list[Chunk]:
    return [Chunk(f"c{i}", f"text {i}", "d.pdf", i, "text", "text", 2) for i in range(n)]


class FakeAsyncLLM:
    def __init__(self, fail: dict[int, list[Exception]] | None = None) -> None:
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls: list[int] = []
        self.fail = fail or {}

    async def acomplete(self, prompt: str) -> str:
        i = int(prompt.split("text ")[-1].split()[0])
        self.calls.append(i)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.005)
            if self.fail.get(i):
                raise self.fail[i].pop(0)
            return TRIPLE_JSON
        finally:
            self.in_flight -= 1


async def no_sleep(_: float) -> None:
    return None


def run_build(llm: FakeAsyncLLM, path: Path, n: int = 12, **kw: Any) -> tuple[Any, list[str]]:
    written: list[str] = []

    def sink(t: Triple, chunk_id: str) -> None:
        written.append(chunk_id)

    stats = asyncio.run(
        build_graph_async(chunks(n), llm, sink, path, sleep=no_sleep, **kw)  # type: ignore[arg-type]
    )
    return stats, written


def test_concurrency_is_bounded(tmp_path: Path) -> None:
    llm = FakeAsyncLLM()
    stats, written = run_build(llm, tmp_path / "ck.json", concurrency=3)
    assert 1 < llm.max_in_flight <= 3
    assert stats.chunks_processed == 12 and len(written) == 12


def test_transient_errors_are_retried(tmp_path: Path) -> None:
    llm = FakeAsyncLLM(fail={4: [RetryableError("429"), RetryableError("429")]})
    stats, _ = run_build(llm, tmp_path / "ck.json")
    assert stats.errors == 0 and stats.chunks_processed == 12
    assert llm.calls.count(4) == 3


def test_permanent_errors_fail_fast_and_are_not_marked_done(tmp_path: Path) -> None:
    path = tmp_path / "ck.json"
    llm = FakeAsyncLLM(fail={5: [ValueError("bad request")]})
    stats, _ = run_build(llm, path)
    assert stats.errors == 1 and llm.calls.count(5) == 1
    assert 5 not in ExtractionCheckpoint.load(path).done

    # Resume: only the failed chunk is attempted again.
    llm2 = FakeAsyncLLM()
    stats2, _ = run_build(llm2, path)
    assert llm2.calls == [5] and stats2.chunks_processed == 1
    assert ExtractionCheckpoint.load(path).done == set(range(12))


def test_start_index_is_still_honoured(tmp_path: Path) -> None:
    llm = FakeAsyncLLM()
    run_build(llm, tmp_path / "ck.json", start_index=8)
    assert sorted(llm.calls) == [8, 9, 10, 11]


def test_reads_checkpoint_written_by_serial_builder(tmp_path: Path) -> None:
    path = tmp_path / "ck.json"
    path.write_text(json.dumps({"chunks_processed": 3, "triples_extracted": 7}))
    ck = ExtractionCheckpoint.load(path)
    assert ck.done == {0, 1, 2} and ck.triples == 7


def test_backoff_delays_grow_and_respect_retry_after() -> None:
    delays: list[float] = []

    async def record(d: float) -> None:
        delays.append(d)

    attempts = {"n": 0}

    async def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 4:
            raise RetryableError("rate limited")
        return "ok"

    out = asyncio.run(call_with_backoff(flaky, sleep=record, rng=random.Random(0)))
    assert out == "ok" and len(delays) == 3
    assert delays[0] < delays[2]  # 1s, 2s, 4s bases with jitter in [0.5, 1.0)

    class Resp:
        headers: ClassVar[dict[str, str]] = {"retry-after": "7"}

    class RateLimitError(Exception):
        response = Resp()

    calls = {"n": 0}

    async def limited() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitError()
        return "ok"

    delays.clear()
    asyncio.run(call_with_backoff(limited, sleep=record))
    assert delays == [7.0]


def test_retry_gives_up_after_limit() -> None:
    async def always() -> str:
        raise RetryableError("still limited")

    with pytest.raises(RetryableError):
        asyncio.run(call_with_backoff(always, retries=2, sleep=no_sleep))


def test_is_retryable_classification() -> None:
    class E(Exception):
        def __init__(self, status: int) -> None:
            self.status_code = status

    assert is_retryable(E(429)) and is_retryable(E(503)) and is_retryable(TimeoutError())
    assert not is_retryable(E(400)) and not is_retryable(ValueError())
