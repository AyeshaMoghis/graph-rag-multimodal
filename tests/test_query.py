"""Query routing, decomposition, HyDE and their wiring into the pipeline."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from graphrag.config import Settings
from graphrag.index.types import ScoredHit
from graphrag.retrieval.hybrid import FusedHit, HybridRetriever
from graphrag.retrieval.pipeline import Pipeline
from graphrag.retrieval.query import (
    TEMPLATES,
    QueryRouter,
    decompose,
    hypothetical_document,
    merge_rankings,
)


class KeywordEncoder:
    """Maps text to one of three axes by keyword, so routing is predictable."""

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for t in texts:
            t = t.lower()
            if any(w in t for w in ("lead", "chain", "combin", "step", "connect", "pipeline")):
                rows.append([0.0, 0.0, 1.0])
            elif any(
                w in t for w in ("differ", "relationship", "compare", "linked", "distinguish")
            ):
                rows.append([0.0, 1.0, 0.0])
            else:
                rows.append([1.0, 0.0, 0.0])
        return np.asarray(rows, dtype=np.float32)


def test_templates_cover_three_classes() -> None:
    assert set(TEMPLATES) == {"lookup", "relational", "multi-hop"}


def test_router_classifies_by_nearest_template() -> None:
    router = QueryRouter(KeywordEncoder())
    assert router.classify("What is politeness detection?") == "lookup"
    assert router.classify("How does toxicity differ from hate speech?") == "relational"
    route = router.route("Trace the full pipeline from signals to a risk score")
    assert (route.query_class, route.graph_hops, route.decompose) == ("multi-hop", 3, True)
    assert router.route("What is MBTI?").use_graph is False


class ScriptedLLM:
    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_decompose_parses_and_caps() -> None:
    llm = ScriptedLLM(json.dumps(["a?", "b?", "c?", "d?"]))
    assert decompose("q", llm) == ["a?", "b?", "c?"]


def test_decompose_and_hyde_fall_back_to_query() -> None:
    assert decompose("q", ScriptedLLM("not json")) == ["q"]
    assert hypothetical_document("q", ScriptedLLM(RuntimeError("down"))) == "q"
    assert hypothetical_document("q", ScriptedLLM("  passage  ")) == "passage"


def fh(cid: str) -> FusedHit:
    return FusedHit(cid, cid, "d", 1, "text", 0.0)


def test_merge_rankings_rewards_chunks_found_by_several_subqueries() -> None:
    merged = merge_rankings([[fh("a"), fh("b")], [fh("c"), fh("b")]], top_k=2)
    assert [h.chunk_id for h in merged] == ["b", "a"]


class RecordingSearcher:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, n: int) -> list[ScoredHit]:
        self.queries.append(query)
        return [ScoredHit(f"{query[:5]}-{i}", f"text {i}", 1.0 - i / 10, {}) for i in range(n)]


class RecordingGraph:
    def __init__(self) -> None:
        self.hops: list[int] = []

    def named_nodes(self) -> list[dict[str, Any]]:
        return [{"name": "MBTI", "types": ["Concept"]}]

    def fulltext(self, text: str, limit: int) -> list[dict[str, Any]]:
        return []

    def expand(
        self, name: str, label: str, hops: int, limit: int, ranked: bool = True
    ) -> list[dict[str, Any]]:
        self.hops.append(hops)
        return []


def make_pipeline(
    llm: Any = None, **settings: Any
) -> tuple[Pipeline, RecordingSearcher, RecordingGraph]:
    dense, sparse, graph = RecordingSearcher(), RecordingSearcher(), RecordingGraph()
    s = Settings(_env_file=None, router=True, **settings)  # type: ignore[call-arg]
    pipeline = Pipeline(
        s,
        HybridRetriever(dense, sparse),
        graph_store=graph,
        llm=llm,
        extract_entities=lambda q: ["MBTI"],
        router=QueryRouter(KeywordEncoder()),
    )
    return pipeline, dense, graph


def test_lookup_route_skips_the_graph() -> None:
    pipeline, _, graph = make_pipeline()
    result = pipeline.retrieve("What is MBTI?")
    assert result.route == "lookup" and graph.hops == []


def test_multi_hop_route_expands_three_hops_and_decomposes() -> None:
    llm = ScriptedLLM(json.dumps(["What is MBTI?", "How does credibility connect?"]))
    pipeline, dense, graph = make_pipeline(llm=llm)
    result = pipeline.retrieve("How does MBTI lead to credibility and then to disinformation?")
    assert result.route == "multi-hop" and graph.hops == [3]
    assert result.sub_queries == ["What is MBTI?", "How does credibility connect?"]
    assert dense.queries == result.sub_queries


def test_hyde_changes_only_the_dense_query() -> None:
    pipeline, dense, _ = make_pipeline(llm=ScriptedLLM("A hypothetical passage."), hyde=True)
    pipeline.router = None
    pipeline.retrieve("What is MBTI?")
    assert dense.queries == ["A hypothetical passage."]
