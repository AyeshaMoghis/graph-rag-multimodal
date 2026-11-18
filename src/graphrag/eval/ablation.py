"""Ablation study (Task 3.2).

Every row is a named set of setting overrides, so each row is reproducible with
``graphrag ablation --only "<row name>"`` (or ``graphrag eval --set k=v ...``).

Methodology
-----------
* Questions: the 50-question golden set.
* Retrieval ground truth: chunk-level relevance groups from
  ``data/golden_qa/relevance_candidates.json``. Reviewed questions use the
  reviewer's labels; unreviewed ones use the automatic suggestions, and the
  table is then marked PROVISIONAL. Identical slides form one group.
* Metrics at k=5 over the retrieved text chunks: Recall@5, MRR (first relevant
  group within the top 5), nDCG@5 with binary gains, and context keyword recall
  (share of expected concepts present in the five chunks).
* Each row runs ``runs`` times (default 3); the table reports mean and standard
  deviation. Retrieval is deterministic (exact score ties break by chunk id),
  so metric spread is expected to be zero; latency spread is not.
* Latency is wall-clock retrieval time per question on the machine that ran
  the study, model loading excluded only insofar as the median ignores the
  first, slower call.
* Answer-level columns (keyword recall on the generated answer, faithfulness)
  need an LLM and are reported as pending until run with a Groq key.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graphrag.config import Settings
from graphrag.eval.harness import GoldenItem
from graphrag.eval.metrics import context_keyword_recall, ndcg_at_k, recall_at_k, reciprocal_rank

ORIGINAL = {
    "fusion": "legacy",
    "bm25_tokenizer": "legacy",
    "graph_mode": "legacy",
    "entity_extractor": "legacy",
    "prompt_version": "v1",
    "conflict_mode": "naive",
}
FIXED = {"fusion": "rrf", "bm25_tokenizer": "normalized"}


@dataclass(frozen=True)
class AblationConfig:
    name: str
    overrides: dict[str, Any]
    needs: tuple[str, ...] = ()  # "graph", "images", "llm"


CONFIGS: tuple[AblationConfig, ...] = (
    AblationConfig("dense only", {**FIXED, "fusion": "dense"}),
    AblationConfig("BM25 only", {**FIXED, "fusion": "sparse"}),
    AblationConfig("hybrid (broken fusion, original)", ORIGINAL),
    AblationConfig("hybrid (RRF, fixed)", FIXED),
    AblationConfig("hybrid + rerank", {**FIXED, "rerank": True}),
    AblationConfig("hybrid + rerank + graph", {**FIXED, "rerank": True}, ("graph",)),
    AblationConfig(
        "full (+ images)", {**FIXED, "rerank": True, "image_top_k": 2}, ("graph", "images")
    ),
)

RETRIEVAL_METRICS = ("recall_at_5", "mrr", "ndcg_at_5", "context_keyword_recall")


@dataclass
class RowResult:
    name: str
    overrides: dict[str, Any]
    status: str  # "measured" | "pending: <reason>"
    runs: list[dict[str, float]] = field(default_factory=list)

    def aggregate(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        if not self.runs:
            return out
        for key in self.runs[0]:
            values = [r[key] for r in self.runs]
            out[key] = {
                "mean": round(statistics.fmean(values), 4),
                "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
            }
        return out


def score_run(
    retrieved: dict[str, list[str]],
    texts: dict[str, list[str]],
    latencies: Sequence[float],
    golden: Sequence[GoldenItem],
    labels: dict[str, dict[str, list[str]]],
) -> dict[str, float]:
    """Mean metrics over questions; questions without relevance groups skip ranking metrics."""
    labelled = [q for q in golden if labels.get(q.id)]
    return {
        "recall_at_5": statistics.fmean(
            recall_at_k(retrieved[q.id], labels[q.id], 5) for q in labelled
        ),
        "mrr": statistics.fmean(
            reciprocal_rank(retrieved[q.id][:5], labels[q.id]) for q in labelled
        ),
        "ndcg_at_5": statistics.fmean(
            ndcg_at_k(retrieved[q.id], labels[q.id], 5) for q in labelled
        ),
        "context_keyword_recall": statistics.fmean(
            context_keyword_recall(texts[q.id], q.expected_concepts) for q in golden
        ),
        "latency_ms_p50": statistics.median(latencies),
        "questions_labelled": float(len(labelled)),
    }


def missing_requirements(config: AblationConfig, settings: Settings) -> list[str]:
    reasons = []
    if "graph" in config.needs and not settings.graph_enabled:
        reasons.append("needs Neo4j (NEO4J_* not set)")
    if "images" in config.needs and not settings.captions_path.exists():
        reasons.append("needs image captions (run `graphrag index` without --skip-images)")
    return reasons


def run_ablation(
    golden: Sequence[GoldenItem],
    labels: dict[str, dict[str, list[str]]],
    base_settings: Settings,
    runs: int = 3,
    only: Sequence[str] | None = None,
) -> list[RowResult]:
    from graphrag.retrieval.pipeline import Pipeline

    rows: list[RowResult] = []
    for config in CONFIGS:
        if only and config.name not in only:
            continue
        settings = base_settings.model_copy(update=config.overrides)
        missing = missing_requirements(config, settings)
        if missing:
            rows.append(RowResult(config.name, config.overrides, "pending: " + "; ".join(missing)))
            continue
        pipeline = Pipeline.from_settings(
            settings, use_graph="graph" in config.needs, use_llm=False
        )
        row = RowResult(config.name, config.overrides, "measured")
        for _ in range(runs):
            retrieved: dict[str, list[str]] = {}
            texts: dict[str, list[str]] = {}
            latencies: list[float] = []
            for q in golden:
                t0 = time.perf_counter()
                hits = pipeline.hybrid.retrieve(q.question, top_k=5)
                latencies.append((time.perf_counter() - t0) * 1000)
                retrieved[q.id] = [h.chunk_id for h in hits]
                texts[q.id] = [h.text for h in hits]
            row.runs.append(score_run(retrieved, texts, latencies, golden, labels))
        rows.append(row)
    return rows


def save_ablation(
    path: Path, rows: Sequence[RowResult], provisional: bool, meta: dict[str, Any]
) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provisional_labels": provisional,
        "meta": meta,
        "rows": [
            {
                "name": r.name,
                "overrides": r.overrides,
                "status": r.status,
                "aggregate": r.aggregate(),
                "runs": r.runs,
            }
            for r in rows
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_table(path: Path) -> str:
    """Markdown table from a saved ablation file (used for the README)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    header = (
        "| Config | Recall@5 | MRR | nDCG@5 | Context keyword recall | "
        "Keyword recall (answer) | Faithfulness | p50 latency |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for row in data["rows"]:
        agg = row["aggregate"]
        if row["status"] != "measured":
            reason = row["status"].removeprefix("pending: ")
            lines.append(f"| {row['name']} | pending | | | | | | {reason} |")
            continue

        def cell(key: str, agg: dict[str, dict[str, float]] = agg) -> str:
            m, sd = agg[key]["mean"], agg[key]["std"]
            return f"{m:.3f}" + (f" ± {sd:.3f}" if sd else "")

        latency = agg["latency_ms_p50"]
        lines.append(
            f"| {row['name']} | {cell('recall_at_5')} | {cell('mrr')} | {cell('ndcg_at_5')} | "
            f"{cell('context_keyword_recall')} | pending (LLM) | pending (LLM) | "
            f"{latency['mean']:.0f} ms ± {latency['std']:.0f} |"
        )
    note = (
        "\n\n*Retrieval labels are automatic suggestions not yet reviewed by a human; "
        "Recall@5, MRR and nDCG@5 are provisional.*"
        if data["provisional_labels"]
        else ""
    )
    meta = data["meta"]
    return (
        "\n".join(lines)
        + note
        + f"\n\n{meta['questions']} questions, {meta['runs']} runs per row, mean ± std "
        "(std omitted when zero)."
    )
