"""Three-layer retrieval orchestration: graph + hybrid text + image captions."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from graphrag.cache import CachedEmbedder, CachedLLM, DiskCache, config_hash
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
from graphrag.reasoning.verify import (
    CrossEncoderNLI,
    NLIModel,
    detect_conflicts,
    evidence_confidence,
)
from graphrag.retrieval.fusion import BudgetConfig, PackedContext, pack_context
from graphrag.retrieval.hybrid import FusedHit, HybridRetriever
from graphrag.retrieval.query import (
    QueryRouter,
    decompose,
    hypothetical_document,
    merge_rankings,
)
from graphrag.retrieval.rerank import CrossEncoderReranker, CrossEncoderScorer

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
    route: str | None = None
    sub_queries: list[str] = field(default_factory=list)
    hyde_passage: str | None = None


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
        nli: NLIModel | None = None,
        router: QueryRouter | None = None,
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
        self.nli: NLIModel | None = nli
        self.router: QueryRouter | None = router
        self.cache: DiskCache | None = None

    @classmethod
    def from_settings(
        cls, settings: Settings, use_graph: bool = True, use_llm: bool = True
    ) -> Pipeline:
        embedder: Any = Embedder(
            settings.embed_model, settings.embed_batch_size, settings.embed_query_cache_size
        )
        cache = DiskCache(settings.cache_path) if settings.cache_enabled else None
        if cache is not None:
            embedder = CachedEmbedder(embedder, cache, config_hash({"m": settings.embed_model}))
        store = VectorStore(settings.chroma_path)
        bm25 = BM25Index.load(settings.bm25_path, tokenizer=settings.bm25_tokenizer)
        hybrid = HybridRetriever(
            dense=DenseSearcher(store, TEXT_COLLECTION, embedder),
            sparse=bm25,
            mode=settings.fusion,
            rrf_k=settings.rrf_k,
            alpha=settings.hybrid_alpha,
            candidate_multiplier=settings.candidate_multiplier,
            reranker=(
                CrossEncoderReranker(CrossEncoderScorer(settings.rerank_model))
                if settings.rerank
                else None
            ),
            rerank_candidates=settings.rerank_candidates,
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
            if cache is not None:
                llm = CachedLLM(
                    llm,
                    cache,
                    config_hash(
                        {
                            "model": settings.llm_model,
                            "temperature": settings.llm_temperature,
                            "max_tokens": settings.llm_max_tokens,
                        }
                    ),
                )
        router = QueryRouter(embedder) if settings.router else None
        pipeline = cls(
            settings, hybrid, image_searcher, graph_store, llm, linker=linker, router=router
        )
        pipeline.cache = cache
        return pipeline

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
        top_k = text_top_k or s.text_top_k

        t0 = time.perf_counter()
        route = self.router.route(query) if self.router is not None else None
        timings["route"] = (time.perf_counter() - t0) * 1000
        if route is not None and graph_hops is None and route.use_graph:
            graph_hops = route.graph_hops
        use_graph = route is None or route.use_graph

        t0 = time.perf_counter()
        graph = GraphResult()
        if self.graph_store is not None and use_graph:
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
        hyde_passage = (
            hypothetical_document(query, self.llm) if s.hyde and self.llm is not None else None
        )
        sub_queries: list[str] = []
        if self.llm is not None and (s.decompose or (route is not None and route.decompose)):
            sub_queries = decompose(query, self.llm)
        if len(sub_queries) > 1:
            texts = merge_rankings(
                [self.hybrid.retrieve(q, top_k=top_k) for q in sub_queries], top_k, s.rrf_k
            )
        else:
            texts = self.hybrid.retrieve(query, top_k=top_k, dense_query=hyde_passage)
        timings["text"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        images = self._images(query, image_top_k if image_top_k is not None else s.image_top_k)
        timings["images"] = (time.perf_counter() - t0) * 1000

        return RetrievalResult(
            query,
            entities,
            graph,
            texts,
            images,
            timings,
            route=route.query_class if route is not None else None,
            sub_queries=sub_queries,
            hyde_passage=hyde_passage,
        )

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
        context = pack_context(
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
        if s.conflict_mode == "nli":
            if self.nli is None:
                self.nli = CrossEncoderNLI(s.nli_model)
            entities = sorted({*result.graph.nodes, *result.entities})
            context.conflicts = [
                c.describe()
                for c in detect_conflicts(
                    context.text_chunks,
                    entities,
                    self.nli,
                    s.nli_threshold,
                    s.nli_max_pairs,
                    encoder=getattr(self.hybrid.dense, "embedder", None),
                    min_similarity=s.nli_min_similarity,
                )
            ]
        return context

    def answer(self, query: str, **retrieve_kwargs: Any) -> tuple[dict[str, Any], RetrievalResult]:
        if self.llm is None:
            raise RuntimeError("GROQ_API_KEY is not set; answering needs an LLM")
        result = self.retrieve(query, **retrieve_kwargs)
        return self.answer_from_retrieval(query, result), result

    def answer_from_retrieval(self, query: str, result: RetrievalResult) -> dict[str, Any]:
        """Pack, generate, verify. Split from :meth:`answer` so the API can stream stages."""
        if self.llm is None:
            raise RuntimeError("GROQ_API_KEY is not set; answering needs an LLM")
        t0 = time.perf_counter()
        context = self.pack(result)
        result.timings_ms["pack"] = (time.perf_counter() - t0) * 1000
        if self.nli is None and self.settings.prompt_version == "v3":
            self.nli = CrossEncoderNLI(self.settings.nli_model)
        t0 = time.perf_counter()
        answer = generate_answer(
            query,
            context,
            self.llm,
            self.settings.prompt_version,
            nli=self.nli if self.settings.prompt_version == "v3" else None,
            entailment_threshold=self.settings.entailment_threshold,
        )
        result.timings_ms["llm"] = (time.perf_counter() - t0) * 1000
        evidence = evidence_confidence(
            result.text_chunks,
            len(context.kept_graph_paths) if self.graph_store is not None else None,
        )
        answer["_usage"] = dict(getattr(self.llm, "last_usage", {}) or {})
        answer["_conflicts"] = list(context.conflicts)
        answer["confidence"] = evidence.level
        answer["_evidence"] = {"score": evidence.score, **evidence.components}
        return answer
