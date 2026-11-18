"""FastAPI endpoints against a fake pipeline (no models, no network)."""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from graphrag.api.app import create_app
from graphrag.config import Settings
from graphrag.graph.retrieve import GraphResult
from graphrag.retrieval.hybrid import FusedHit
from graphrag.retrieval.pipeline import RetrievalResult


class FakeGraph:
    def named_nodes(self) -> list[dict[str, Any]]:
        return [{"name": "MBTI", "types": ["Concept"]}]

    def fulltext(self, text: str, limit: int) -> list[dict[str, Any]]:
        return [{"name": "MBTI", "types": ["Concept"], "score": 1.0}]

    def expand(self, name: str, label: str, hops: int, limit: int, ranked: bool = True) -> list:
        return [
            {
                "names": ["MBTI", "Personality Model"],
                "rel_types": ["IS_A"],
                "dirs": ["->"],
                "end_name": "Personality Model",
                "end_type": ["Concept"],
                "hops": 1,
                "support": 1,
            }
        ]


class FakePipeline:
    def __init__(self, llm: bool = True, graph: bool = False) -> None:
        self.llm = object() if llm else None
        self.graph_store = FakeGraph() if graph else None
        self.linker = None
        self.cache = None

    def retrieve(self, query: str, **kw: Any) -> RetrievalResult:
        hit = FusedHit("c1", "Credibility is a node property.", "lec.pdf", 13, "slide", 0.03, 1, 1)
        return RetrievalResult(query, ["MBTI"], GraphResult(), [hit], [], {"text": 1.0})

    def answer_from_retrieval(self, query: str, result: RetrievalResult) -> dict[str, Any]:
        return {
            "answer": "Credibility is a node property [c1].",
            "source_chunks_used": ["c1"],
            "confidence": "high",
            "_usage": {"prompt_tokens": 100, "completion_tokens": 20},
            "_conflicts": [],
        }

    def answer(self, query: str, **kw: Any) -> tuple[dict[str, Any], RetrievalResult]:
        result = self.retrieve(query)
        return self.answer_from_retrieval(query, result), result


def client(**kw: Any) -> TestClient:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    app = create_app(lambda s: FakePipeline(**kw), settings=settings)
    return TestClient(app)


def test_health_and_request_id_echo() -> None:
    with client() as c:
        r = c.get("/health", headers={"X-Request-ID": "abc123"})
        assert r.status_code == 200 and r.json()["llm"] is True
        assert r.headers["X-Request-ID"] == "abc123"
        assert len(c.get("/health").headers["X-Request-ID"]) == 32  # generated


def test_ask_json() -> None:
    with client() as c:
        r = c.post("/ask", json={"question": "What is credibility?", "stream": False})
        body = r.json()
        assert r.status_code == 200
        assert body["retrieval"]["chunks"][0]["chunk_id"] == "c1"
        assert body["retrieval"]["chunks"][0]["retrieved_by"] == ["dense", "sparse"]
        assert body["answer"]["confidence"] == "high"
        assert body["usage"] == {"prompt_tokens": 100, "completion_tokens": 20}


def test_ask_streams_stages_in_order() -> None:
    with client() as c, c.stream("POST", "/ask", json={"question": "What is credibility?"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [line.split(": ", 1)[1] for line in r.iter_lines() if line.startswith("event:")]
    assert events == ["retrieval", "answer", "done"]


def test_ask_without_llm_is_503() -> None:
    with client(llm=False) as c:
        r = c.post("/ask", json={"question": "What is credibility?", "stream": False})
        assert r.status_code == 503 and "GROQ_API_KEY" in r.json()["detail"]


def test_validation_errors_are_422() -> None:
    with client() as c:
        assert c.post("/ask", json={"question": "x"}).status_code == 422
        assert c.post("/ask", json={"question": "valid?", "graph_hops": 9}).status_code == 422


def test_subgraph() -> None:
    with client() as c:
        assert c.get("/graph/subgraph", params={"entity": "MBTI"}).status_code == 503
    with client(graph=True) as c:
        r = c.get("/graph/subgraph", params={"entity": "mbti", "hops": 1})
        assert r.status_code == 200
        assert r.json()["paths"] == ["MBTI -[IS_A]-> Personality Model"]


def test_sse_payloads_are_json() -> None:
    with client() as c, c.stream("POST", "/ask", json={"question": "What is credibility?"}) as r:
        data = [line.split(": ", 1)[1] for line in r.iter_lines() if line.startswith("data:")]
    assert json.loads(data[0])["chunks"][0]["doc_id"] == "lec.pdf"
    assert json.loads(data[-1])["usage"]["prompt_tokens"] == 100
