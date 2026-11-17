"""Three-layer retrieval orchestration: graph + hybrid text + image captions."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from graphrag.config import Settings
from graphrag.graph.retrieve import (
    EntityExtractor,
    EntityLinker,
    GlinerNER,
    GraphResult,
    GraphStore,
    extract_entities_simple,
    graph_retrieve,
)
from graphrag.index.embeddings import Embedder
from graphrag.index.sparse import BM25Index
from graphrag.index.vector_store import (
    IMAGE_COLLECTION,
    TEXT_COLLECTION,
    DenseSearcher,
    VectorStore,
)
from graphrag.log import get_logger
from graphrag.reasoning.answer import generate_answer
from graphrag.reasoning.llm import LLM
from graphrag.retrieval.fusion import BudgetConfig, PackedContext, pack_context
from graphrag.retrieval.hybrid import FusedHit, HybridRetriever

log = get_logger(__name__)


@dataclass
class ImageHit:
    image_id: str
    caption: str
    image_path: str
    doc_id: str
    page: int
    score: float


@dataclass
class RetrievalResult:
    query: str
    entities: list[str]
    graph: GraphResult
    text_chunks: list[FusedHit]
    images: list[ImageHit]
    timings_ms: dict[str, float] = field(default_factory=dict)


class Pipeline:
    """Owns the retrievers. Construct with :meth:`from_settings` or inject parts in tests."""

    def __init__(
        self,
        settings: Settings,
        hybrid: HybridRetriever,
        image_searcher: DenseSearcher | None = None,
        graph_store: GraphStore | None = None,
        llm: LLM | None = None,
        extract_entities: Callable[[str], list[str]] | None = None,
        linker: EntityLinker | None = None,
    ) -> None:
        self.settings = settings
        self.hybrid = hybrid
        self.image_searcher = image_searcher
        self.graph_store = graph_store
        self.llm = llm
        if extract_entities is None:
            extract_entities = (
                extract_entities_simple
                if settings.entity_extractor == "legacy"
                else EntityExtractor(
                    GlinerNER(settings.ner_model, threshold=settings.ner_threshold)
                )
            )
        self.extract_entities = extract_entities
        self.linker = linker

    @classmethod
    def from_settings(
        cls, settings: Settings, use_graph: bool = True, use_llm: bool = True
    ) -> Pipeline:
        embedder = Embedder(settings.embed_model, settings.embed_batch_size)
        store = VectorStore(settings.chroma_path)
        bm25 = BM25Index.load(settings.bm25_path, tokenizer=settings.bm25_tokenizer)
        hybrid = HybridRetriever(
            dense=DenseSearcher(store, TEXT_COLLECTION, embedder),
            sparse=bm25,
            mode=settings.fusion,
            rrf_k=settings.rrf_k,
            alpha=settings.hybrid_alpha,
            candidate_multiplier=settings.candidate_multiplier,
        )
        image_searcher = DenseSearcher(store, IMAGE_COLLECTION, embedder)

        graph_store: GraphStore | None = None
        linker: EntityLinker | None = None
        if use_graph and settings.graph_enabled:
            from graphrag.graph.retrieve import Neo4jGraphStore
            from graphrag.graph.schema import connect

            graph_store = Neo4jGraphStore(connect(settings))
            linker = EntityLinker(
                graph_store,
                encoder=embedder,
                lexical_threshold=settings.link_lexical_threshold,
                embedding_threshold=settings.link_embedding_threshold,
                fuzzy_threshold=settings.graph_fuzzy_threshold,
                cache_dir=settings.embeddings_dir,
            )
        elif use_graph:
            log.warning("graph_disabled", reason="NEO4J_* not configured")

        llm: LLM | None = None
        if use_llm and settings.groq_api_key is not None:
            from graphrag.reasoning.llm import GroqLLM

            llm = GroqLLM(
                settings.groq_api_key.get_secret_value(),
                settings.llm_model,
                settings.llm_temperature,
                settings.llm_max_tokens,
            )
        return cls(settings, hybrid, image_searcher, graph_store, llm, linker=linker)

    def retrieve(
        self,
        query: str,
        text_top_k: int | None = None,
        image_top_k: int | None = None,
        graph_hops: int | None = None,
    ) -> RetrievalResult:
        s = self.settings
        timings: dict[str, float] = {}
        entities: list[str] = []

        t0 = time.perf_counter()
        graph = GraphResult()
        if self.graph_store is not None:
            entities = self.extract_entities(query)
            graph = graph_retrieve(
                entities,
                self.graph_store,
                hops=graph_hops or s.graph_hops,
                top_n_entities=s.graph_top_n_entities,
                fuzzy_threshold=s.graph_fuzzy_threshold,
                path_limit=s.graph_path_limit,
                mode=s.graph_mode,
                linker=self.linker,
            )
        timings["graph"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        texts = self.hybrid.retrieve(query, top_k=text_top_k or s.text_top_k)
        timings["text"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        images = self._images(query, image_top_k if image_top_k is not None else s.image_top_k)
        timings["images"] = (time.perf_counter() - t0) * 1000

        return RetrievalResult(query, entities, graph, texts, images, timings)

    def _images(self, query: str, top_k: int) -> list[ImageHit]:
        if self.image_searcher is None or top_k <= 0:
            return []
        try:
            hits = self.image_searcher.search(query, top_k * 2)
        except Exception as exc:  # collection missing when a corpus has no images
            log.warning("image_search_failed", error=str(exc))
            return []
        hits.sort(key=lambda h: h.score, reverse=True)
        return [
            ImageHit(
                image_id=h.chunk_id,
                caption=h.text,
                image_path=str(h.metadata.get("image_path", "")),
                doc_id=str(h.metadata.get("doc_id", "")),
                page=int(h.metadata.get("page", 0)),
                score=h.score,
            )
            for h in hits[:top_k]
        ]

    def pack(self, result: RetrievalResult) -> PackedContext:
        s = self.settings
        return pack_context(
            result.graph.paths,
            result.graph.nodes,
            result.text_chunks,
            result.images,
            BudgetConfig(
                s.context_budget_tokens,
                s.graph_budget_fraction,
                s.text_budget_fraction,
                s.image_budget_fraction,
            ),
        )

    def answer(self, query: str, **retrieve_kwargs: Any) -> tuple[dict[str, Any], RetrievalResult]:
        if self.llm is None:
            raise RuntimeError("GROQ_API_KEY is not set; answering needs an LLM")
        result = self.retrieve(query, **retrieve_kwargs)
        t0 = time.perf_counter()
        answer = generate_answer(query, self.pack(result), self.llm)
        result.timings_ms["llm"] = (time.perf_counter() - t0) * 1000
        return answer, result
