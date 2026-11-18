"""Fusion scoring, including the regression test for the original key-mismatch bug."""

from __future__ import annotations

import pytest

from graphrag.index.types import ScoredHit
from graphrag.retrieval.hybrid import (
    HybridRetriever,
    fuse,
    min_max_normalize,
    reciprocal_rank_fusion,
    weighted_fusion,
)


def hit(cid: str, score: float, text: str | None = None) -> ScoredHit:
    return ScoredHit(
        cid, text or f"text of {cid} " + "x" * 40, score, {"doc_id": "d.pdf", "page": 1}
    )


def score_of(results: list, cid: str) -> float:
    return next(r.score for r in results if r.chunk_id == cid)


# ── The regression test the brief asks for ──────────────────────────────────


@pytest.mark.parametrize("mode", ["rrf", "weighted"])
def test_chunk_found_by_both_retrievers_outscores_same_chunk_found_by_one(mode: str) -> None:
    """A chunk at the same ranks must score strictly higher when both retrievers agree."""
    dense = [hit("a", 0.90), hit("b", 0.80), hit("c", 0.70)]
    sparse_with_b = [hit("b", 12.0), hit("d", 8.0), hit("e", 3.0)]
    sparse_without_b = [hit("x", 12.0), hit("d", 8.0), hit("e", 3.0)]

    both = fuse(dense, sparse_with_b, mode=mode)  # type: ignore[arg-type]
    one = fuse(dense, sparse_without_b, mode=mode)  # type: ignore[arg-type]

    assert score_of(both, "b") > score_of(one, "b")
    b = next(r for r in both if r.chunk_id == "b")
    assert b.retrieved_by == ("dense", "sparse")


def test_rrf_puts_consensus_chunk_first() -> None:
    dense = [hit("a", 0.9), hit("b", 0.8)]
    sparse = [hit("c", 20.0), hit("b", 10.0)]
    assert fuse(dense, sparse, mode="rrf")[0].chunk_id == "b"


def test_legacy_mode_reproduces_the_bug() -> None:
    """Pins the original behaviour so the 'before' number is measured honestly.

    Dense hits were keyed on text[:30] and BM25 hits on chunk_id, so a chunk
    returned by both retrievers enters fusion twice and never gets both signals.
    """
    dense = [hit("b", 0.8)]
    sparse = [hit("b", 10.0)]
    results = fuse(dense, sparse, mode="legacy")

    assert [r.chunk_id for r in results] == ["b", "b"]  # same chunk occupies two slots
    assert all(len(r.retrieved_by) == 1 for r in results)
    assert {r.fusion_key for r in results} == {"b", dense[0].text[:30]}


def test_legacy_mode_collapses_chunks_sharing_a_30_char_prefix() -> None:
    same_prefix = "Lecture 4: Social Network Analysis overview"
    dense = [hit("p1", 0.9, same_prefix + " part one"), hit("p2", 0.8, same_prefix + " part two")]
    results = fuse(dense, [], mode="legacy")
    assert len(results) == 1


# ── Building blocks ─────────────────────────────────────────────────────────


def test_rrf_formula() -> None:
    scores = reciprocal_rank_fusion([["a", "b"], ["b"]], k=60)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_rejects_negative_k() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=-1)


def test_min_max_handles_constant_and_empty() -> None:
    assert min_max_normalize({}) == {}
    assert min_max_normalize({"a": 3.0, "b": 3.0}) == {"a": 1.0, "b": 1.0}
    assert min_max_normalize({"a": 0.0}) == {"a": 0.0}
    assert min_max_normalize({"a": 1.0, "b": 3.0}) == {"a": 0.0, "b": 1.0}


def test_weighted_fusion_alpha_bounds() -> None:
    with pytest.raises(ValueError):
        weighted_fusion({"a": 1.0}, {}, alpha=1.5)


def test_fuse_is_deterministic_on_ties() -> None:
    dense = [hit("z", 0.5), hit("y", 0.5)]
    sparse = [hit("y", 1.0), hit("z", 1.0)]
    first = [r.chunk_id for r in fuse(dense, sparse)]
    assert first == [r.chunk_id for r in fuse(dense, sparse)]


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        fuse([hit("a", 1.0)], [], mode="bogus")  # type: ignore[arg-type]


class FakeSearcher:
    def __init__(self, hits: list[ScoredHit]) -> None:
        self.hits = hits
        self.requested: list[int] = []

    def search(self, query: str, n: int) -> list[ScoredHit]:
        self.requested.append(n)
        return self.hits[:n]


def test_retriever_over_retrieves_and_truncates() -> None:
    dense = FakeSearcher([hit(f"d{i}", 1 - i / 100) for i in range(30)])
    sparse = FakeSearcher([hit(f"s{i}", 30 - i) for i in range(30)])
    out = HybridRetriever(dense, sparse, candidate_multiplier=3).retrieve("q", top_k=5)
    assert len(out) == 5
    assert dense.requested == [15] and sparse.requested == [15]


class ReverseScorer:
    """Scores later candidates higher, so reranking visibly reorders."""

    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.pairs = list(pairs)
        return [float(i) for i in range(len(pairs))]


def test_reranker_over_retrieves_and_reorders() -> None:
    from graphrag.retrieval.rerank import CrossEncoderReranker

    dense = FakeSearcher([hit(f"d{i}", 1 - i / 100) for i in range(40)])
    sparse = FakeSearcher([hit(f"s{i}", 40 - i) for i in range(40)])
    scorer = ReverseScorer()
    retriever = HybridRetriever(
        dense, sparse, reranker=CrossEncoderReranker(scorer), rerank_candidates=30
    )
    out = retriever.retrieve("q", top_k=5)
    assert len(scorer.pairs) == 30  # the cross-encoder saw the 30-candidate pool
    assert dense.requested == [30] and sparse.requested == [30]
    assert [h.rerank_score for h in out] == [29.0, 28.0, 27.0, 26.0, 25.0]


def test_single_retriever_modes_query_only_that_retriever() -> None:
    dense = FakeSearcher([hit(f"d{i}", 1 - i / 100) for i in range(20)])
    sparse = FakeSearcher([hit(f"s{i}", 20 - i) for i in range(20)])
    out = HybridRetriever(dense, sparse, mode="dense").retrieve("q", top_k=3)
    assert [h.chunk_id for h in out] == ["d0", "d1", "d2"] and sparse.requested == []
    out = HybridRetriever(dense, sparse, mode="sparse").retrieve("q", top_k=3)
    assert [h.chunk_id for h in out] == ["s0", "s1", "s2"]
