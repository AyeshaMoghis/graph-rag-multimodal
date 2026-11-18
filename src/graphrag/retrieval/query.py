"""Query understanding (Task 3.4): routing, decomposition and HyDE.

Router
    Zero-shot: each class is described by generic template questions, and a
    query is assigned to the class whose nearest template is most similar in
    embedding space. Nothing is trained on the golden labels. The templates
    were written by someone who had seen the golden questions, so accuracy
    against those labels is indicative, not a held-out score.

Routes
    lookup     -> no graph layer
    relational -> graph expansion at 2 hops
    multi-hop  -> graph expansion at 3 hops, plus query decomposition

Decomposition
    The LLM splits a multi-hop question into at most three sub-questions; each
    is retrieved separately and the rankings are merged with RRF.

HyDE
    The LLM writes a short hypothetical answer; dense retrieval embeds that
    passage instead of the question. BM25 still uses the question.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from graphrag.graph.extract import strip_fences
from graphrag.reasoning.llm import LLM

QueryClass = Literal["lookup", "relational", "multi-hop"]

TEMPLATES: dict[QueryClass, tuple[str, ...]] = {
    "lookup": (
        "What is the definition of this term?",
        "What are the types of this thing?",
        "List the components of this method.",
        "What does this concept mean?",
        "Which factors are used to measure this?",
    ),
    "relational": (
        "How does one concept differ from another concept?",
        "What is the relationship between these two ideas?",
        "Compare these two approaches.",
        "How is the first thing linked to the second thing?",
        "What distinguishes one category from the other?",
    ),
    "multi-hop": (
        "How does the first factor lead to the second, and how does that in turn affect a third?",
        "Design a system that combines several techniques, step by step, and explain why.",
        "Trace the full chain from the initial signals to the final outcome.",
        "What combination of methods would best solve this complex problem, and why?",
        "How do several concepts connect to each other to produce an overall effect?",
    ),
}


@dataclass(frozen=True)
class Route:
    query_class: QueryClass
    use_graph: bool
    graph_hops: int
    decompose: bool


ROUTES: dict[QueryClass, Route] = {
    "lookup": Route("lookup", use_graph=False, graph_hops=0, decompose=False),
    "relational": Route("relational", use_graph=True, graph_hops=2, decompose=False),
    "multi-hop": Route("multi-hop", use_graph=True, graph_hops=3, decompose=True),
}

# Golden-set question types mapped onto router classes, for measuring accuracy.
GOLDEN_TO_CLASS: dict[str, QueryClass] = {
    "factual": "lookup",
    "relational": "relational",
    "multi-hop": "multi-hop",
}


class QueryRouter:
    def __init__(self, encoder: Any) -> None:
        self.encoder = encoder
        self._classes: list[QueryClass] = []
        self._matrix: np.ndarray | None = None

    def _templates(self) -> np.ndarray:
        if self._matrix is None:
            texts: list[str] = []
            for cls, templates in TEMPLATES.items():
                self._classes.extend([cls] * len(templates))
                texts.extend(templates)
            self._matrix = np.asarray(self.encoder.encode(texts), dtype=np.float32)
        return self._matrix

    def scores(self, query: str) -> dict[QueryClass, float]:
        sims = self._templates() @ np.asarray(self.encoder.encode([query]), dtype=np.float32)[0]
        best: dict[QueryClass, float] = {}
        for cls, sim in zip(self._classes, sims, strict=True):
            best[cls] = max(best.get(cls, -1.0), float(sim))
        return best

    def classify(self, query: str) -> QueryClass:
        s = self.scores(query)
        return max(s, key=lambda c: s[c])

    def route(self, query: str) -> Route:
        return ROUTES[self.classify(query)]


DECOMPOSE_PROMPT = """Break the question into at most three short, self-contained sub-questions
that together cover everything needed to answer it. Return ONLY a JSON array of
strings, no explanation.

QUESTION: REPLACE_QUERY_HERE
"""

HYDE_PROMPT = """Write a short factual passage (3 to 4 sentences) that would answer the
question below, in the style of lecture notes. Return only the passage.

QUESTION: REPLACE_QUERY_HERE
"""


def decompose(query: str, llm: LLM, max_parts: int = 3) -> list[str]:
    """Sub-questions from the LLM; falls back to the original query on any failure."""
    try:
        parts = json.loads(
            strip_fences(llm.complete(DECOMPOSE_PROMPT.replace("REPLACE_QUERY_HERE", query)))
        )
        subs = [str(p).strip() for p in parts if str(p).strip()] if isinstance(parts, list) else []
    except Exception:
        subs = []
    return subs[:max_parts] or [query]


def hypothetical_document(query: str, llm: LLM) -> str:
    """HyDE passage; falls back to the query itself on failure."""
    try:
        passage = llm.complete(HYDE_PROMPT.replace("REPLACE_QUERY_HERE", query)).strip()
    except Exception:
        passage = ""
    return passage or query


def merge_rankings(rankings: Sequence[Sequence[Any]], top_k: int, k: int = 60) -> list[Any]:
    """RRF across sub-question rankings, keyed on chunk id; keeps each chunk's first hit object."""
    from graphrag.retrieval.hybrid import reciprocal_rank_fusion

    scores = reciprocal_rank_fusion([[h.chunk_id for h in r] for r in rankings], k=k)
    first: dict[str, Any] = {}
    for ranking in rankings:
        for h in ranking:
            first.setdefault(h.chunk_id, h)
    order = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return [first[cid] for cid in order[:top_k]]
