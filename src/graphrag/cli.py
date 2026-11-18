"""Command-line interface: ``graphrag ingest | index | build-graph | ask | eval``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from graphrag.config import Settings
from graphrag.log import configure_logging, get_logger

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
log = get_logger("graphrag.cli")

PRESETS: dict[str, dict[str, str]] = {
    # Reproduces the original notebook's broken hybrid retrieval exactly.
    "original": {
        "fusion": "legacy",
        "bm25_tokenizer": "legacy",
        "graph_mode": "legacy",
        "entity_extractor": "legacy",
        "prompt_version": "v1",
        "conflict_mode": "naive",
    },
    # The fix: RRF over true chunk ids with normalised BM25 tokens.
    "fixed": {"fusion": "rrf", "bm25_tokenizer": "normalized"},
}


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    configure_logging("DEBUG" if verbose else "INFO")


def parse_overrides(items: list[str]) -> dict[str, str]:
    """Parse ``key=value`` pairs; pydantic converts the strings to the field types."""
    out: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise typer.BadParameter(f"expected key=value, got {item!r}")
        out[key.strip()] = value.strip()
    return out


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
    start_index: int = typer.Option(0, help="Skip chunks before this index."),
    limit: int | None = typer.Option(None, help="Only process this many chunks."),
    concurrency: int = typer.Option(4, help="Concurrent LLM calls."),
    serial: bool = typer.Option(False, help="Use the original serial loop with a fixed sleep."),
) -> None:
    """Extract triples with the LLM and upsert them into Neo4j. Safe to re-run: finished
    chunks are skipped and failed chunks are retried."""
    import asyncio

    from graphrag.graph.extract import build_graph, build_graph_async, upsert_triple
    from graphrag.graph.schema import apply_schema, connect, graph_stats
    from graphrag.ingest.chunking import load_chunks
    from graphrag.reasoning.llm import AsyncGroqLLM, GroqLLM

    s = _settings()
    if not s.graph_enabled or s.groq_api_key is None:
        raise typer.BadParameter(
            "NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD and GROQ_API_KEY are required"
        )
    key = s.groq_api_key.get_secret_value()
    driver = connect(s)
    apply_schema(driver)
    chunks = load_chunks(s.documents_dir)
    if limit is not None:
        chunks = chunks[: start_index + limit]
    if serial:
        stats = build_graph(
            chunks,
            GroqLLM(key, s.llm_model, s.llm_temperature, s.llm_max_tokens),
            driver,
            s.graph_checkpoint_path,
            start_index,
            s.triple_confidence_threshold,
            s.extraction_sleep_seconds,
            s.extraction_checkpoint_every,
        )
    else:
        with driver.session() as session:
            stats = asyncio.run(
                build_graph_async(
                    chunks,
                    AsyncGroqLLM(key, s.llm_model, s.llm_temperature, s.llm_max_tokens),
                    lambda t, cid: upsert_triple(session, t, cid),
                    s.graph_checkpoint_path,
                    start_index=start_index,
                    concurrency=concurrency,
                    threshold=s.triple_confidence_threshold,
                    char_limit=s.triple_text_char_limit,
                )
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
    overrides: list[str] = typer.Option(
        [], "--set", help="Override any setting, e.g. --set rerank=true (repeatable)."
    ),
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
    config_overrides: dict[str, object] = dict(PRESETS[preset]) if preset else {}
    if fusion:
        config_overrides["fusion"] = fusion
    if tokenizer:
        config_overrides["bm25_tokenizer"] = tokenizer
    config_overrides.update(parse_overrides(overrides))
    s = _settings(**config_overrides)

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
        "rerank": s.rerank,
        "rerank_model": s.rerank_model if s.rerank else None,
        "rerank_candidates": s.rerank_candidates if s.rerank else None,
        "llm_model": s.llm_model if mode == "answer" else None,
        "golden_qa": str(s.golden_qa_path),
        "n_questions": len(golden),
        "overrides": config_overrides,
    }
    save_run(out, config, summary, results)
    out.with_suffix(".md").write_text(render_markdown(summary, config), encoding="utf-8")
    typer.echo(json.dumps(summary, indent=2))


@app.command()
def calibration(
    results: Path = typer.Argument(..., help="An answer-mode results JSON from `graphrag eval`."),
    out: Path = typer.Option(Path("docs/calibration.png"), help="Where to save the plot."),
) -> None:
    """Plot confidence bucket against keyword recall, for self-reported and evidence confidence."""
    from graphrag.eval.calibration import plot_calibration

    typer.echo(json.dumps(plot_calibration(results, out), indent=2))


@app.command()
def conflicts(
    out: Path = typer.Option(Path("results/conflicts.json"), help="Where to save the comparison."),
    limit: int | None = typer.Option(None, help="Only the first N questions."),
    gates: str = typer.Option("0.0,0.6,0.7,0.8", help="Relatedness gates (cosine) to report."),
) -> None:
    """Compare the old substring check with the NLI check, at several relatedness gates.

    Corpus: both methods see the same top-5 retrieved chunks and the same entity
    set, the entities extracted from each golden question. (The notebook used
    graph node names; that needs Neo4j and is not used here.) Planted set:
    recall on synthetic contradictions and false positives on paraphrases.
    """
    from datetime import datetime, timezone

    from graphrag.eval.conflict_eval import evaluate_planted
    from graphrag.eval.harness import load_golden
    from graphrag.graph.retrieve import EntityExtractor, GlinerNER
    from graphrag.index.embeddings import Embedder
    from graphrag.reasoning.verify import CrossEncoderNLI, candidate_pairs, detect_conflicts
    from graphrag.retrieval.fusion import naive_conflicts
    from graphrag.retrieval.pipeline import Pipeline

    s = _settings()
    pipeline = Pipeline.from_settings(s, use_graph=False, use_llm=False)
    extract = EntityExtractor(GlinerNER(s.ner_model, threshold=s.ner_threshold))
    nli = CrossEncoderNLI(s.nli_model)
    encoder = Embedder(s.embed_model, s.embed_batch_size)
    gate_values = [float(g) for g in gates.split(",")]

    questions = []
    for item in load_golden(s.golden_qa_path)[:limit]:
        hits = pipeline.hybrid.retrieve(item.question, top_k=s.text_top_k)
        questions.append((item.id, hits, extract(item.question)))

    naive = {qid: naive_conflicts(hits, ents) for qid, hits, ents in questions}
    sweep: list[dict[str, Any]] = []
    for gate in gate_values:
        flagged = {
            qid: [
                c.__dict__
                for c in detect_conflicts(
                    hits, ents, nli, s.nli_threshold, s.nli_max_pairs, encoder, gate
                )
            ]
            for qid, hits, ents in questions
        }
        sweep.append(
            {
                "min_similarity": gate,
                "corpus_conflicts": sum(len(v) for v in flagged.values()),
                "corpus_questions_flagged": sum(1 for v in flagged.values() if v),
                "planted": evaluate_planted(
                    Path("data/golden_qa/planted_conflicts.json"),
                    nli,
                    encoder,
                    s.nli_threshold,
                    gate,
                ),
                "flagged": {k: v for k, v in flagged.items() if v},
            }
        )
    summary = {
        "questions": len(questions),
        "naive_conflicts": sum(len(v) for v in naive.values()),
        "naive_questions_flagged": sum(1 for v in naive.values() if v),
        "candidate_pairs": sum(
            len(candidate_pairs(h, e, s.nli_max_pairs)) for _, h, e in questions
        ),
        "nli_model": s.nli_model,
        "nli_threshold": s.nli_threshold,
        "entity_source": "question entities (GLiNER + regex); graph nodes not used",
        "sweep": [{k: v for k, v in row.items() if k != "flagged"} for row in sweep],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "summary": summary,
                "entities": {qid: ents for qid, _, ents in questions},
                "naive": {k: v for k, v in naive.items() if v},
                "sweep": sweep,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command("label-candidates")
def label_candidates(
    out: Path = typer.Option(
        Path("data/golden_qa/relevance_candidates.json"), help="Where to write the candidates."
    ),
    force: bool = typer.Option(False, help="Overwrite even if some questions are reviewed."),
) -> None:
    """Generate chunk-level relevance candidates for manual review (Task 3.3)."""
    from graphrag.eval.relevance import LABEL_MODEL, generate_candidates
    from graphrag.index.embeddings import Embedder
    from graphrag.ingest.chunking import load_chunks

    if out.exists() and not force:
        existing = json.loads(out.read_text(encoding="utf-8"))
        if any(q.get("reviewed") for q in existing.get("questions", [])):
            raise typer.BadParameter(f"{out} has reviewed questions; use --force to overwrite")
    s = _settings()
    questions = json.loads(s.golden_qa_path.read_text(encoding="utf-8"))
    data = generate_candidates(questions, load_chunks(s.documents_dir), Embedder(LABEL_MODEL))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    suggested = [sum(c["suggested"] for c in q["candidates"]) for q in data["questions"]]
    typer.echo(
        f"questions {len(suggested)} | with >=1 suggestion {sum(1 for n in suggested if n)} | "
        f"suggested groups total {sum(suggested)} | status: UNREVIEWED"
    )


@app.command()
def ablation(
    runs: int = typer.Option(3, help="Runs per row."),
    only: list[str] = typer.Option([], help="Run only these rows (repeatable)."),
    out: Path = typer.Option(Path("results/ablation.json"), help="Where to save raw results."),
    labels: Path = typer.Option(
        Path("data/golden_qa/relevance_candidates.json"), help="Relevance labels file."
    ),
) -> None:
    """Run the ablation table (Task 3.2) and save raw results plus a markdown table."""
    from graphrag.eval.ablation import render_table, run_ablation, save_ablation
    from graphrag.eval.harness import load_golden
    from graphrag.eval.relevance import load_relevance

    s = _settings()
    golden = load_golden(s.golden_qa_path)
    relevance, reviewed = load_relevance(labels)
    rows = run_ablation(golden, relevance, s, runs=runs, only=only or None)
    provisional = not all(reviewed.get(q.id) for q in golden)
    save_ablation(
        out, rows, provisional, {"questions": len(golden), "runs": runs, "labels": str(labels)}
    )
    table = render_table(out)
    out.with_suffix(".md").write_text(table + "\n", encoding="utf-8")
    typer.echo(table)


if __name__ == "__main__":
    app()
