"""Cross-encoder reranking between fusion and context packing (Task 3.1).

Fusion over-retrieves ``rerank_candidates`` chunks (default 30); a
cross-encoder scores each (query, chunk) pair jointly and the top-k by that
score are kept. A bi-encoder embeds query and chunk separately, so it cannot
model their interaction; the cross-encoder can, at the cost of one forward pass
per candidate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from graphrag.retrieval.hybrid import FusedHit


class Reranker(Protocol):
    def rerank(self, query: str, hits: Sequence[FusedHit], top_k: int) -> list[FusedHit]: ...


class PairScorer(Protocol):
    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...


class CrossEncoderScorer:
    """Lazy wrapper around a sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", max_length: int = 512) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._model: Any = None

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, max_length=self.max_length)
        return [float(s) for s in self._model.predict(list(pairs), show_progress_bar=False)]


class CrossEncoderReranker:
    def __init__(self, scorer: PairScorer) -> None:
        self.scorer = scorer

    def rerank(self, query: str, hits: Sequence[FusedHit], top_k: int) -> list[FusedHit]:
        """Order by cross-encoder score; ties keep the fused order, so output is deterministic."""
        scores = self.scorer.score([(query, h.text) for h in hits])
        order = sorted(range(len(hits)), key=lambda i: (-round(scores[i], 6), i))
        return [replace(hits[i], rerank_score=scores[i]) for i in order[:top_k]]
