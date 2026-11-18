"""Golden-QA evaluation harness with checkpointing, summaries and reports.

Modes
-----
``answer``     Full pipeline with the LLM. Reports the original keyword recall on
               the generated answer plus the original (self-reported) graph
               utilisation and confidence, so numbers compare to the baseline.
``retrieval``  Text retrieval only, no LLM and no graph. Reports keyword recall
               over the retrieved chunks. This isolates the fusion fix.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from graphrag.eval.judge import judge_answer
from graphrag.eval.metrics import context_keyword_recall, keyword_recall
from graphrag.log import get_logger
from graphrag.reasoning.llm import LLM
from graphrag.reasoning.verify import evidence_confidence
from graphrag.retrieval.pipeline import Pipeline

log = get_logger(__name__)

EvalMode = Literal["answer", "retrieval"]


@dataclass(frozen=True)
class GoldenItem:
    id: str
    question: str
    question_type: str
    expected_concepts: list[str]


def load_golden(path: Path) -> list[GoldenItem]:
    if not path.exists():
        raise FileNotFoundError(f"Golden QA set not found at {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        GoldenItem(
            id=str(r["id"]),
            question=str(r["question"]),
            question_type=str(r.get("question_type", "factual")),
            expected_concepts=list(r.get("expected_concepts", [])),
        )
        for r in rows
    ]


def evaluate_item(
    pipeline: Pipeline, item: GoldenItem, mode: EvalMode, judge: LLM | None = None
) -> dict[str, Any]:
    s = pipeline.settings
    row: dict[str, Any] = {
        "id": item.id,
        "question": item.question,
        "question_type": item.question_type,
        "expected_concepts": item.expected_concepts,
    }
    t0 = time.perf_counter()
    if mode == "retrieval":
        hits = pipeline.hybrid.retrieve(item.question, top_k=s.text_top_k)
        ids = [h.chunk_id for h in hits]
        row.update(
            context_keyword_recall=context_keyword_recall(
                [h.text for h in hits], item.expected_concepts
            ),
            retrieved_chunk_ids=ids,
            duplicate_slots=len(ids) - len(set(ids)),
            both_signal_hits=sum(1 for h in hits if len(h.retrieved_by) == 2),
            evidence_confidence=evidence_confidence(hits, None).level,
        )
    else:
        answer, result = pipeline.answer(item.question, text_top_k=5, image_top_k=2, graph_hops=2)
        meta = answer.get("_fusion_metadata", {})
        citations = answer.get("_citations", {})
        kept, dropped = citations.get("kept", {}), citations.get("dropped", {})
        claimed_paths = kept.get("graph_paths_used", []) + dropped.get("graph_paths_used", [])
        row.update(
            answer=answer.get("answer", ""),
            keyword_recall=keyword_recall(str(answer.get("answer", "")), item.expected_concepts),
            context_keyword_recall=context_keyword_recall(
                [h.text for h in result.text_chunks], item.expected_concepts
            ),
            # Original definition: the model *claims* it used a graph path.
            graph_utilized=len(claimed_paths) > 0,
            graph_paths_count=len(claimed_paths),
            # Claims that survived citation validation.
            graph_utilized_validated=len(kept.get("graph_paths_used", [])) > 0,
            # What was actually in the prompt, for comparison.
            graph_paths_in_context=meta.get("graph_paths_in_context", 0),
            confidence=answer.get("confidence", "low"),
            confidence_self_reported=answer.get("confidence_self_reported") or "missing",
            evidence=answer.get("_evidence", {}),
            citations_cited=citations.get("cited", 0),
            citations_fabricated=citations.get("fabricated", 0),
            citations_dropped=dropped,
            supported_sentence_rate=answer.get("supported_sentence_rate"),
            unsupported_claims=answer.get("unsupported_claims", []),
            referenced_concepts=answer.get("referenced_concepts", []),
            retrieved_chunk_ids=[h.chunk_id for h in result.text_chunks],
            text_chunks_used=meta.get("text_chunks_used", 0),
            images_used=meta.get("images_used", 0),
            timings_ms=result.timings_ms,
        )
        if judge is not None:
            context = "\n".join([*result.graph.paths, *(h.text for h in result.text_chunks)])
            verdict = judge_answer(item.question, context, row["answer"], judge)
            row.update(
                judge_faithfulness=verdict.faithfulness,
                judge_answer_relevance=verdict.answer_relevance,
                judge_error=verdict.error,
            )
    row["latency_ms"] = (time.perf_counter() - t0) * 1000
    return row


def run_evaluation(
    pipeline: Pipeline,
    golden: Sequence[GoldenItem],
    mode: EvalMode,
    checkpoint_path: Path | None = None,
    start_from: int = 0,
    sleep_seconds: float = 0.0,
    judge: LLM | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if start_from and checkpoint_path and checkpoint_path.exists():
        results = json.loads(checkpoint_path.read_text(encoding="utf-8"))[:start_from]
    for i, item in enumerate(golden[start_from:], start=start_from):
        try:
            row = evaluate_item(pipeline, item, mode, judge)
        except Exception as exc:
            log.error("eval_item_failed", id=item.id, error=str(exc))
            row = {
                "id": item.id,
                "question": item.question,
                "question_type": item.question_type,
                "error": str(exc),
            }
        results.append(row)
        log.info(
            "eval_item",
            n=i + 1,
            total=len(golden),
            id=item.id,
            recall=row.get("keyword_recall", row.get("context_keyword_recall")),
        )
        if checkpoint_path and (i + 1) % 10 == 0:
            _write(checkpoint_path, results)
        if sleep_seconds and mode == "answer":
            time.sleep(sleep_seconds)
    return results


def summarize(results: Sequence[dict[str, Any]], mode: EvalMode) -> dict[str, Any]:
    good = [r for r in results if "error" not in r]
    metric = "keyword_recall" if mode == "answer" else "context_keyword_recall"

    def mean(rows: Sequence[dict[str, Any]], key: str) -> float:
        return round(sum(float(r[key]) for r in rows) / len(rows), 3) if rows else 0.0

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in good:
        by_type[r["question_type"]].append(r)

    summary: dict[str, Any] = {
        "mode": mode,
        "total": len(results),
        "successful": len(good),
        "errors": len(results) - len(good),
        "primary_metric": metric,
        metric: mean(good, metric),
        "latency_ms_p50": (
            round(statistics.median(r["latency_ms"] for r in good), 1) if good else None
        ),
        "by_type": {
            t: {"n": len(rows), metric: mean(rows, metric)} for t, rows in sorted(by_type.items())
        },
    }
    if mode == "answer":
        summary["context_keyword_recall"] = mean(good, "context_keyword_recall")
        summary["graph_utilization_self_reported"] = mean(
            [{"v": float(r["graph_utilized"])} for r in good], "v"
        )
        summary["graph_paths_in_context_rate"] = mean(
            [{"v": float(r["graph_paths_in_context"] > 0)} for r in good], "v"
        )
        summary["graph_utilization_validated"] = mean(
            [{"v": float(r["graph_utilized_validated"])} for r in good], "v"
        )
        cited = sum(r["citations_cited"] for r in good)
        fabricated = sum(r["citations_fabricated"] for r in good)
        summary["citations_total"] = cited
        summary["citations_fabricated"] = fabricated
        summary["fabricated_citation_rate"] = round(fabricated / cited, 3) if cited else 0.0
        summary["answers_with_fabrication_rate"] = mean(
            [{"v": float(r["citations_fabricated"] > 0)} for r in good], "v"
        )
        judged = [
            r for r in good if r.get("judge_faithfulness") is not None and not r.get("judge_error")
        ]
        if judged:
            summary["judge_faithfulness"] = mean(judged, "judge_faithfulness")
            summary["judge_answer_relevance"] = mean(judged, "judge_answer_relevance")
            summary["judge_errors"] = sum(1 for r in good if r.get("judge_error"))
        rated = [r for r in good if r.get("supported_sentence_rate") is not None]
        summary["supported_sentence_rate"] = (
            mean(rated, "supported_sentence_rate") if rated else None
        )
        summary["confidence"] = dict(Counter(r["confidence"] for r in good))
        summary["confidence_self_reported"] = dict(
            Counter(r["confidence_self_reported"] for r in good)
        )
        summary["calibration"] = {
            source: {
                level: {"n": len(rows), "keyword_recall": mean(rows, "keyword_recall")}
                for level in ("high", "medium", "low")
                if (rows := [r for r in good if r[source] == level])
            }
            for source in ("confidence_self_reported", "confidence")
        }
        for t, rows in by_type.items():
            summary["by_type"][t]["graph_self_reported"] = mean(
                [{"v": float(r["graph_utilized"])} for r in rows], "v"
            )
    else:
        summary["duplicate_slots_total"] = sum(r["duplicate_slots"] for r in good)
        summary["both_signal_hits_total"] = sum(r["both_signal_hits"] for r in good)
        summary["evidence_confidence_text_only"] = dict(
            Counter(r.get("evidence_confidence", "n/a") for r in good)
        )
    return summary


def save_run(
    path: Path, config: dict[str, Any], summary: dict[str, Any], results: Sequence[dict[str, Any]]
) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "config": config,
        "summary": summary,
        "results": list(results),
    }
    _write(path, payload)


def render_markdown(summary: dict[str, Any], config: dict[str, Any]) -> str:
    metric = summary["primary_metric"]
    lines = [
        "# Evaluation report",
        "",
        f"- Mode: `{summary['mode']}`",
        f"- Fusion: `{config.get('fusion')}`, BM25 tokenizer: `{config.get('bm25_tokenizer')}`",
        f"- Questions: {summary['total']} ({summary['errors']} errors)",
        f"- **{metric}: {summary[metric]:.3f}**",
        f"- p50 latency: {summary['latency_ms_p50']} ms",
        "",
        "| Type | n | " + metric + " |",
        "|---|---|---|",
    ]
    for t, row in summary["by_type"].items():
        lines.append(f"| {t} | {row['n']} | {row[metric]:.3f} |")
    return "\n".join(lines) + "\n"


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
