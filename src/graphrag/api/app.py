"""FastAPI service (Task 3.6).

Endpoints
---------
``POST /ask``            Answer a question. Streams Server-Sent Events by stage
                         (``retrieval`` -> ``answer`` -> ``done``); pass
                         ``"stream": false`` for a single JSON response.
``POST /ingest``         Re-run extraction, chunking and text indexing.
``GET  /graph/subgraph`` Paths around an entity, as used for answers.
``GET  /health``         Index, graph, LLM and cache status.

Every response carries an ``X-Request-ID`` header (taken from the request or
generated), and every request logs per-stage latency and LLM token usage.

The stream is per stage rather than per token: the answer is JSON that must be
parsed and citation-checked before it is safe to show, so partial tokens would
expose unvalidated citations.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from graphrag.config import Settings, get_settings
from graphrag.log import configure_logging, get_logger

log = get_logger("graphrag.api")


# ── Request / response models ───────────────────────────────────────────────


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    text_top_k: int | None = Field(default=None, ge=1, le=20)
    image_top_k: int | None = Field(default=None, ge=0, le=10)
    graph_hops: int | None = Field(default=None, ge=1, le=3)
    stream: bool = True


class ChunkOut(BaseModel):
    chunk_id: str
    doc_id: str
    page: int
    text: str
    score: float
    retrieved_by: list[str]


class ImageOut(BaseModel):
    image_id: str
    caption: str
    doc_id: str
    page: int
    score: float


class RetrievalOut(BaseModel):
    route: str | None
    sub_queries: list[str]
    entities: list[str]
    graph_paths: list[str]
    chunks: list[ChunkOut]
    images: list[ImageOut]


class AnswerOut(BaseModel):
    answer: str
    sentences: list[dict[str, Any]] = []
    unsupported_claims: list[str] = []
    source_chunks_used: list[str] = []
    graph_paths_used: list[str] = []
    relevant_images: list[str] = []
    confidence: str
    confidence_self_reported: str | None = None
    citations: dict[str, Any] = {}
    conflicts: list[str] = []


class AskResponse(BaseModel):
    request_id: str
    retrieval: RetrievalOut
    answer: AnswerOut
    timings_ms: dict[str, float]
    usage: dict[str, int]


class IngestRequest(BaseModel):
    corpus_dir: str | None = None


class SubgraphOut(BaseModel):
    entity: str
    linked: list[dict[str, Any]]
    paths: list[str]
    nodes: list[str]


# ── Helpers ─────────────────────────────────────────────────────────────────


def retrieval_out(result: Any) -> RetrievalOut:
    return RetrievalOut(
        route=result.route,
        sub_queries=result.sub_queries,
        entities=result.entities,
        graph_paths=result.graph.paths,
        chunks=[
            ChunkOut(
                chunk_id=h.chunk_id,
                doc_id=h.doc_id,
                page=h.page,
                text=h.text,
                score=h.rerank_score if h.rerank_score is not None else h.score,
                retrieved_by=list(h.retrieved_by),
            )
            for h in result.text_chunks
        ],
        images=[
            ImageOut(
                image_id=i.image_id, caption=i.caption, doc_id=i.doc_id, page=i.page, score=i.score
            )
            for i in result.images
        ],
    )


def answer_out(answer: dict[str, Any], conflicts: list[str]) -> AnswerOut:
    return AnswerOut(
        answer=str(answer.get("answer", "")),
        sentences=answer.get("sentences", []),
        unsupported_claims=answer.get("unsupported_claims", []),
        source_chunks_used=answer.get("source_chunks_used", []),
        graph_paths_used=answer.get("graph_paths_used", []),
        relevant_images=answer.get("relevant_images", []),
        confidence=str(answer.get("confidence", "low")),
        confidence_self_reported=answer.get("confidence_self_reported"),
        citations=answer.get("_citations", {}),
        conflicts=conflicts,
    )


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


# ── Application ─────────────────────────────────────────────────────────────


def create_app(
    pipeline_factory: Callable[[Settings], Any] | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Build the app. Tests inject ``pipeline_factory`` to avoid loading models."""

    def default_factory(s: Settings) -> Any:
        from graphrag.retrieval.pipeline import Pipeline

        return Pipeline.from_settings(s)

    factory = pipeline_factory or default_factory
    state: dict[str, Any] = {"settings": settings or get_settings(), "pipeline": None}
    ingest_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        configure_logging(json=True)
        state["pipeline"] = factory(state["settings"])
        yield

    app = FastAPI(title="graph-rag-multimodal", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_id_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("request_id", "path")
        response.headers["X-Request-ID"] = request_id
        log.info("request", request_id=request_id, ms=round((time.perf_counter() - t0) * 1000, 1))
        return response

    def pipeline() -> Any:
        if state["pipeline"] is None:
            raise HTTPException(503, "pipeline not initialised")
        return state["pipeline"]

    @app.get("/health")
    def health() -> dict[str, Any]:
        p = pipeline()
        cache = getattr(p, "cache", None)
        s: Settings = state["settings"]
        return {
            "status": "ok",
            "graph": p.graph_store is not None,
            "llm": p.llm is not None,
            "fusion": s.fusion,
            "rerank": s.rerank,
            "cache": cache.stats.as_dict(s.usd_per_million_tokens) if cache else None,
        }

    def run_ask(req: AskRequest, request_id: str) -> tuple[Any, dict[str, Any], list[str]]:
        p = pipeline()
        if p.llm is None:
            raise HTTPException(503, "GROQ_API_KEY is not set; /ask needs an LLM")
        answer, result = p.answer(
            req.question,
            text_top_k=req.text_top_k,
            image_top_k=req.image_top_k,
            graph_hops=req.graph_hops,
        )
        conflicts = answer.get("_conflicts", [])
        log.info(
            "ask",
            request_id=request_id,
            timings_ms={k: round(v, 1) for k, v in result.timings_ms.items()},
            usage=answer.get("_usage", {}),
            route=result.route,
        )
        return result, answer, conflicts

    @app.post("/ask", response_model=None)
    async def ask(req: AskRequest, request: Request) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        if not req.stream:
            result, answer, conflicts = await asyncio.to_thread(run_ask, req, request_id)
            body = AskResponse(
                request_id=request_id,
                retrieval=retrieval_out(result),
                answer=answer_out(answer, conflicts),
                timings_ms=result.timings_ms,
                usage=answer.get("_usage", {}),
            )
            return JSONResponse(body.model_dump())

        p = pipeline()
        if p.llm is None:
            raise HTTPException(503, "GROQ_API_KEY is not set; /ask needs an LLM")

        async def events() -> AsyncIterator[str]:
            try:
                result = await asyncio.to_thread(
                    p.retrieve,
                    req.question,
                    text_top_k=req.text_top_k,
                    image_top_k=req.image_top_k,
                    graph_hops=req.graph_hops,
                )
                yield sse("retrieval", retrieval_out(result).model_dump())
                answer = await asyncio.to_thread(p.answer_from_retrieval, req.question, result)
                yield sse("answer", answer_out(answer, answer.get("_conflicts", [])).model_dump())
                yield sse(
                    "done",
                    {
                        "request_id": request_id,
                        "timings_ms": result.timings_ms,
                        "usage": answer.get("_usage", {}),
                    },
                )
            except Exception as exc:  # surface failures to the client as an event
                log.error("ask_failed", request_id=request_id, error=str(exc))
                yield sse("error", {"request_id": request_id, "detail": str(exc)})

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"X-Request-ID": request_id}
        )

    @app.get("/graph/subgraph", response_model=SubgraphOut)
    def subgraph(
        entity: str = Query(min_length=1, max_length=200),
        hops: int = Query(2, ge=1, le=3),
    ) -> SubgraphOut:
        from graphrag.graph.retrieve import graph_retrieve

        p = pipeline()
        if p.graph_store is None:
            raise HTTPException(503, "graph layer disabled: NEO4J_* not configured")
        s: Settings = state["settings"]
        result = graph_retrieve(
            [entity],
            p.graph_store,
            hops=hops,
            top_n_entities=1,
            path_limit=s.graph_path_limit,
            mode=s.graph_mode,
            linker=p.linker,
        )
        return SubgraphOut(
            entity=entity,
            linked=[asdict(link) for link in result.links],
            paths=result.paths,
            nodes=result.nodes,
        )

    @app.post("/ingest")
    def ingest(req: IngestRequest) -> dict[str, Any]:
        if not ingest_lock.acquire(blocking=False):
            raise HTTPException(409, "an ingest is already running")
        try:
            from graphrag.index.embeddings import Embedder
            from graphrag.index.sparse import BM25Index
            from graphrag.index.vector_store import TEXT_COLLECTION, VectorStore
            from graphrag.ingest.chunking import ChunkingConfig, chunk_pages, save_chunks
            from graphrag.ingest.pdf_extract import extract_corpus, load_pages

            s: Settings = state["settings"]
            if req.corpus_dir:
                s = s.model_copy(update={"corpus_dir": req.corpus_dir})
            stats = extract_corpus(s.corpus_dir, s.documents_dir, s.phash_threshold)
            chunks = chunk_pages(
                load_pages(s.documents_dir),
                ChunkingConfig(
                    s.chunk_min_words, s.chunk_dense_line_count, s.chunk_min_chunk_words
                ),
            )
            save_chunks(chunks, s.documents_dir)
            vectors = Embedder(s.embed_model, s.embed_batch_size).encode([c.text for c in chunks])
            VectorStore(s.chroma_path).rebuild(
                TEXT_COLLECTION,
                [c.chunk_id for c in chunks],
                [c.text for c in chunks],
                vectors,
                [
                    {"doc_id": c.doc_id, "page": c.page, "page_type": c.page_type, "source": "text"}
                    for c in chunks
                ],
            )
            BM25Index(
                [c.chunk_id for c in chunks],
                [c.text for c in chunks],
                [{"doc_id": c.doc_id, "page": c.page, "page_type": c.page_type} for c in chunks],
            ).save(s.bm25_path)
            state["pipeline"] = factory(s)
            return {
                "pdfs": stats.pdfs,
                "pages": stats.pages,
                "images_kept": stats.images_kept,
                "chunks": len(chunks),
                "note": "image captions are rebuilt with `graphrag index`, not here",
            }
        finally:
            ingest_lock.release()

    return app


def app_factory() -> FastAPI:
    """Entry point for ``uvicorn graphrag.api.app:app_factory --factory``."""
    return create_app()
