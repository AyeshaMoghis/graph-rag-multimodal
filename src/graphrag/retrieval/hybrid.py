"""Hybrid dense + BM25 retrieval.

Three fusion modes:

``rrf`` (default)
    Reciprocal Rank Fusion: ``score(d) = sum_r 1 / (k + rank_r(d))`` with k=60.
    Uses ranks only, so it is immune to the scale mismatch between cosine
    similarity and unbounded BM25 scores.

``weighted``
    ``alpha * dense + (1 - alpha) * bm25`` after min-max normalising each
    retriever's candidate list. Kept as a configurable alternative.

``legacy``
    A faithful reproduction of the original notebook, kept only so the bug can
    be measured. Dense hits were keyed by ``meta.get("chunk_id", doc[:30])``,
    but ``chunk_id`` was never written to Chroma metadata, so every dense key
    was the first 30 characters of the text. BM25 hits were keyed by the real
    chunk id. The key spaces never intersect, so no chunk ever received both
    signals. Dense scores were raw similarities and BM25 scores were divided by
    the top BM25 score.

Every mode fuses on true chunk ids except ``legacy``, which deliberately uses
the broken keys. :class:`FusedHit` always records the true ``chunk_id`` so that
retrieval metrics can be computed for every mode.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from graphrag.config import FusionMode
from graphrag.index.types import ScoredHit, Searcher
from graphrag.retrieval.rerank import Reranker

LEGACY_KEY_CHARS = 30


@dataclass
class FusedHit:
    chunk_id: str
    text: str
    doc_id: str
    page: int
    page_type: str
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_key: str = ""
    rerank_score: float | None = None

    @property
    def retrieved_by(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, rank in (("dense", self.dense_rank), ("sparse", self.sparse_rank))
            if rank is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]:
    """Sum of ``1 / (k + rank)`` over every ranking an id appears in. Ranks start at 1."""
    if k < 0:
        raise ValueError("k must be non-negative")
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] += 1.0 / (k + rank)
    return dict(scores)


def min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {key: (1.0 if hi > 0 else 0.0) for key in scores}
    return {key: (value - lo) / (hi - lo) for key, value in scores.items()}


def weighted_fusion(
    dense: dict[str, float], sparse: dict[str, float], alpha: float = 0.5
) -> dict[str, float]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    d, s = min_max_normalize(dense), min_max_normalize(sparse)
    return {
        key: alpha * d.get(key, 0.0) + (1 - alpha) * s.get(key, 0.0) for key in d.keys() | s.keys()
    }


def legacy_dense_key(hit: ScoredHit) -> str:
    """The key the original code effectively used for dense hits (see module docstring)."""
    return hit.text[:LEGACY_KEY_CHARS]


def _legacy_scores(
    dense_hits: Sequence[ScoredHit], sparse_hits: Sequence[ScoredHit], alpha: float
) -> tuple[dict[str, float], dict[str, str], dict[str, str]]:
    """Reproduce the original scoring. Returns (scores, dense_key->chunk_id, sparse_key->chunk_id)."""
    dense_scores: dict[str, float] = {}
    dense_ids: dict[str, str] = {}
    for hit in dense_hits:
        key = legacy_dense_key(hit)
        dense_scores[key] = hit.score  # later hits with the same prefix overwrite earlier ones
        dense_ids[key] = hit.chunk_id

    top = sparse_hits[0].score if sparse_hits else 0.0
    max_bm = top if top > 0 else 1.0
    sparse_scores = {hit.chunk_id: hit.score / max_bm for hit in sparse_hits}
    sparse_ids = {hit.chunk_id: hit.chunk_id for hit in sparse_hits}

    combined = {
        key: alpha * dense_scores.get(key, 0.0) + (1 - alpha) * sparse_scores.get(key, 0.0)
        for key in dense_scores.keys() | sparse_scores.keys()
    }
    return combined, dense_ids, sparse_ids


def fuse(
    dense_hits: Sequence[ScoredHit],
    sparse_hits: Sequence[ScoredHit],
    mode: FusionMode = "rrf",
    rrf_k: int = 60,
    alpha: float = 0.5,
) -> list[FusedHit]:
    """Fuse two ranked lists into one list sorted by fused score, best first.

    Ties break on the best rank across retrievers, then on chunk id, so results
    are deterministic.
    """
    dense_rank = {h.chunk_id: i for i, h in enumerate(dense_hits, start=1)}
    sparse_rank = {h.chunk_id: i for i, h in enumerate(sparse_hits, start=1)}
    dense_by_id = {h.chunk_id: h for h in dense_hits}
    sparse_by_id = {h.chunk_id: h for h in sparse_hits}

    key_to_id: dict[str, str]
    if mode == "rrf":
        scores = reciprocal_rank_fusion(
            [[h.chunk_id for h in dense_hits], [h.chunk_id for h in sparse_hits]], k=rrf_k
        )
        key_to_id = {cid: cid for cid in scores}
    elif mode == "weighted":
        scores = weighted_fusion(
            {h.chunk_id: h.score for h in dense_hits},
            {h.chunk_id: h.score for h in sparse_hits},
            alpha=alpha,
        )
        key_to_id = {cid: cid for cid in scores}
    elif mode == "legacy":
        scores, dense_ids, sparse_ids = _legacy_scores(dense_hits, sparse_hits, alpha)
        key_to_id = {**sparse_ids, **dense_ids}
    else:
        raise ValueError(f"unknown fusion mode: {mode}")

    fused: list[FusedHit] = []
    for key, score in scores.items():
        cid = key_to_id[key]
        source = dense_by_id.get(cid) or sparse_by_id[cid]
        meta = source.metadata
        # In legacy mode a chunk only "has" a signal if it entered fusion under that key.
        in_dense = cid in dense_rank and (mode != "legacy" or key != cid)
        in_sparse = cid in sparse_rank and (mode != "legacy" or key == cid)
        fused.append(
            FusedHit(
                chunk_id=cid,
                text=source.text,
                doc_id=str(meta.get("doc_id", "")),
                page=int(meta.get("page", 0)),
                page_type=str(meta.get("page_type", "")),
                score=score,
                dense_rank=dense_rank[cid] if in_dense else None,
                sparse_rank=sparse_rank[cid] if in_sparse else None,
                dense_score=dense_by_id[cid].score if in_dense else None,
                sparse_score=sparse_by_id[cid].score if in_sparse else None,
                fusion_key=key,
            )
        )

    def best_rank(h: FusedHit) -> int:
        return min(r for r in (h.dense_rank, h.sparse_rank, 10**9) if r is not None)

    fused.sort(key=lambda h: (-h.score, best_rank(h), h.chunk_id))
    return fused


class HybridRetriever:
    def __init__(
        self,
        dense: Searcher,
        sparse: Searcher,
        mode: FusionMode = "rrf",
        rrf_k: int = 60,
        alpha: float = 0.5,
        candidate_multiplier: int = 3,
        reranker: Reranker | None = None,
        rerank_candidates: int = 30,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.mode: FusionMode = mode
        self.rrf_k = rrf_k
        self.alpha = alpha
        self.candidate_multiplier = candidate_multiplier
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates

    def retrieve(
        self, query: str, top_k: int = 5, dense_query: str | None = None
    ) -> list[FusedHit]:
        """Fuse, then optionally rerank the top ``rerank_candidates`` with a cross-encoder.

        ``dense`` and ``sparse`` modes query only that retriever (for the
        ablation); a single list fused with RRF keeps that retriever's order.
        ``dense_query`` replaces the text sent to the dense retriever (HyDE);
        BM25 and the reranker always see the original query.
        """
        dq = dense_query or query
        pool = max(top_k, self.rerank_candidates) if self.reranker else top_k
        n = max(top_k * self.candidate_multiplier, pool)
        if self.mode == "dense":
            hits = fuse(self.dense.search(dq, n), [], mode="rrf", rrf_k=self.rrf_k)
        elif self.mode == "sparse":
            hits = fuse([], self.sparse.search(query, n), mode="rrf", rrf_k=self.rrf_k)
        else:
            hits = fuse(
                self.dense.search(dq, n),
                self.sparse.search(query, n),
                mode=self.mode,
                rrf_k=self.rrf_k,
                alpha=self.alpha,
            )
        if self.reranker is not None:
            return self.reranker.rerank(query, hits[:pool], top_k)
        return hits[:top_k]
