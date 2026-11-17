"""Command-line interface: ``graphrag ingest | index | build-graph | ask | eval``."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from graphrag.config import Settings
from graphrag.log import configure_logging, get_logger

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
log = get_logger("graphrag.cli")

PRESETS: dict[str, dict[str, str]] = {
    # Reproduces the original notebook's broken hybrid retrieval exactly.
    "original": {"fusion": "legacy", "bm25_tokenizer": "legacy"},
    # The fix: RRF over true chunk ids with normalised BM25 tokens.
    "fixed": {"fusion": "rrf", "bm25_tokenizer": "normalized"},
}


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    configure_logging("DEBUG" if verbose else "INFO")


def _settings(**overrides: object) -> Settings:
    clean = {k: v for k, v in overrides.items() if v is not None}
    return Settings(**clean)  # type: ignore[arg-type]


@app.command()
def ingest(corpus_dir: Path | None = typer.Option(None, help="Folder of PDFs.")) -> None:
    """Extract text and images from PDFs, dedupe images, and chunk pages."""
    from graphrag.ingest.chunking import ChunkingConfig, chunk_pages, save_chunks
    from graphrag.ingest.pdf_extract import extract_corpus, load_pages

    s = _settings(corpus_dir=corpus_dir)
    stats = extract_corpus(s.corpus_dir, s.documents_dir, s.phash_threshold)
    chunks = chunk_pages(
        load_pages(s.documents_dir),
        ChunkingConfig(s.chunk_min_words, s.chunk_dense_line_count, s.chunk_min_chunk_words),
    )
    save_chunks(chunks, s.documents_dir)
    typer.echo(
        f"PDFs {stats.pdfs} | pages {stats.pages} | images seen {stats.images_seen} | "
        f"duplicates {stats.duplicates} | images kept {stats.images_kept} | chunks {len(chunks)}"
    )


@app.command()
def index(skip_images: bool = typer.Option(False, help="Skip OCR/BLIP captioning.")) -> None:
    """Embed chunks into Chroma, build BM25, and caption + embed images."""
    from dataclasses import asdict

    from PIL import Image

    from graphrag.index.captions import Captioner, CaptionRecord, image_id_for, is_useless
    from graphrag.index.embeddings import Embedder
    from graphrag.index.sparse import BM25Index
    from graphrag.index.vector_store import IMAGE_COLLECTION, TEXT_COLLECTION, VectorStore
    from graphrag.ingest.chunking import load_chunks
    from graphrag.ingest.pdf_extract import load_image_records, write_json

    s = _settings()
    chunks = load_chunks(s.documents_dir)
    embedder = Embedder(s.embed_model, s.embed_batch_size)
    store = VectorStore(s.chroma_path)

    vectors = embedder.encode([c.text for c in chunks], show_progress=True)
    n_text = store.rebuild(
        TEXT_COLLECTION,
        [c.chunk_id for c in chunks],
        [c.text for c in chunks],
        vectors,
        [
            {
                "doc_id": c.doc_id,
                "page": c.page,
                "page_type": c.page_type,
                "source": "text",
                "word_count": c.word_count,
            }
            for c in chunks
        ],
    )
    metas = [
        {"doc_id": c.doc_id, "page": c.page, "page_type": c.page_type, "source": "text"}
        for c in chunks
    ]
    BM25Index([c.chunk_id for c in chunks], [c.text for c in chunks], metas).save(s.bm25_path)
    typer.echo(f"text_chunks {n_text} | bm25 {len(chunks)}")

    if skip_images:
        return
    captioner = Captioner(s.caption_model, s.ocr_min_alpha_words)
    captions: list[CaptionRecord] = []
    counts = {"ocr": 0, "blip": 0, "useless": 0}
    for rec in load_image_records(s.documents_dir):
        if not Path(rec.image_path).exists():
            continue
        with Image.open(rec.image_path) as img:
            text, method = captioner.caption(img.convert("RGB"))
        counts[method] += 1
        if is_useless(text, s.caption_min_words):
            counts["useless"] += 1
            continue
        captions.append(
            CaptionRecord(
                image_id_for(rec.doc_id, rec.image_path),
                rec.image_path,
                text,
                method,
                rec.doc_id,
                rec.page,
            )
        )
    write_json(s.captions_path, [asdict(c) for c in captions])
    n_img = 0
    if captions:
        n_img = store.rebuild(
            IMAGE_COLLECTION,
            [c.image_id for c in captions],
            [c.caption for c in captions],
            embedder.encode([c.caption for c in captions]),
            [
                {
                    "image_path": c.image_path,
                    "doc_id": c.doc_id,
                    "page": c.page,
                    "method": c.method,
                    "source": "image",
                }
                for c in captions
            ],
        )
    typer.echo(
        f"ocr {counts['ocr']} | blip {counts['blip']} | useless {counts['useless']} | "
        f"image_chunks {n_img}"
    )


@app.command("build-graph")
def build_graph_cmd(
    start_index: int = typer.Option(0, help="Resume from this chunk index."),
    limit: int | None = typer.Option(None, help="Only process this many chunks."),
) -> None:
    """Extract triples with the LLM and upsert them into Neo4j."""

    from graphrag.graph.extract import build_graph
    from graphrag.graph.schema import apply_schema, connect, graph_stats
    from graphrag.ingest.chunking import load_chunks
    from graphrag.reasoning.llm import GroqLLM

    s = _settings()
    if not s.graph_enabled or s.groq_api_key is None:
        raise typer.BadParameter(
            "NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD and GROQ_API_KEY are required"
        )
    driver = connect(s)
    apply_schema(driver)
    chunks = load_chunks(s.documents_dir)
    if limit is not None:
        chunks = chunks[: start_index + limit]
    llm = GroqLLM(
        s.groq_api_key.get_secret_value(), s.llm_model, s.llm_temperature, s.llm_max_tokens
    )
    stats = build_graph(
        chunks,
        llm,
        driver,
        s.graph_checkpoint_path,
        start_index,
        s.triple_confidence_threshold,
        s.extraction_sleep_seconds,
        s.extraction_checkpoint_every,
    )
    typer.echo(f"chunks {stats.chunks_processed} | triples {stats.triples} | errors {stats.errors}")
    typer.echo(json.dumps(graph_stats(driver), indent=2))
    driver.close()


@app.command()
def ask(
    question: str,
    fusion: str | None = typer.Option(None, help="rrf | weighted | legacy"),
) -> None:
    """Answer a question with the full three-layer pipeline."""
    from graphrag.retrieval.pipeline import Pipeline

    s = _settings(fusion=fusion)
    answer, result = Pipeline.from_settings(s).answer(question)
    typer.echo(json.dumps({k: v for k, v in answer.items() if not k.startswith("_")}, indent=2))
    typer.echo(f"chunks: {[h.chunk_id for h in result.text_chunks]}")
    typer.echo(f"graph paths in context: {len(result.graph.paths)} | timings: {result.timings_ms}")


@app.command("eval")
def eval_cmd(
    mode: str = typer.Option("answer", help="answer (needs GROQ_API_KEY) | retrieval (no LLM)"),
    preset: str | None = typer.Option(None, help="original | fixed"),
    fusion: str | None = typer.Option(None, help="rrf | weighted | legacy"),
    tokenizer: str | None = typer.Option(None, help="normalized | legacy"),
    out: Path = typer.Option(Path("results/eval.json"), help="Where to save raw results."),
    limit: int | None = typer.Option(None, help="Only the first N questions."),
    start_from: int = typer.Option(0, help="Resume from question index."),
) -> None:
    """Run the golden QA set and save raw results, a summary and a markdown report."""
    from graphrag.eval.harness import (
        load_golden,
        render_markdown,
        run_evaluation,
        save_run,
        summarize,
    )
    from graphrag.retrieval.pipeline import Pipeline

    if mode not in ("answer", "retrieval"):
        raise typer.BadParameter("mode must be answer or retrieval")
    overrides: dict[str, object] = dict(PRESETS[preset]) if preset else {}
    if fusion:
        overrides["fusion"] = fusion
    if tokenizer:
        overrides["bm25_tokenizer"] = tokenizer
    s = _settings(**overrides)

    golden = load_golden(s.golden_qa_path)[:limit]
    pipeline = Pipeline.from_settings(s, use_graph=mode == "answer", use_llm=mode == "answer")
    results = run_evaluation(
        pipeline,
        golden,
        mode,  # type: ignore[arg-type]
        checkpoint_path=out.with_suffix(".partial.json"),
        start_from=start_from,
        sleep_seconds=s.eval_sleep_seconds,
    )
    summary = summarize(results, mode)  # type: ignore[arg-type]
    config = {
        "preset": preset,
        "fusion": s.fusion,
        "bm25_tokenizer": s.bm25_tokenizer,
        "rrf_k": s.rrf_k,
        "alpha": s.hybrid_alpha,
        "text_top_k": s.text_top_k,
        "embed_model": s.embed_model,
        "llm_model": s.llm_model if mode == "answer" else None,
        "golden_qa": str(s.golden_qa_path),
        "n_questions": len(golden),
    }
    save_run(out, config, summary, results)
    out.with_suffix(".md").write_text(render_markdown(summary, config), encoding="utf-8")
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
