"""Post-hoc checks on generated answers.

* :func:`validate_citations` drops any citation that does not point at something
  that was actually in the prompt, and counts it as fabricated.
* :func:`evidence_confidence` scores confidence from retrieval evidence instead
  of trusting the model's self-report.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from graphrag.reasoning.prompts import graph_path_id
from graphrag.retrieval.fusion import PackedContext

CITATION_FIELDS = ("source_chunks_used", "graph_paths_used", "relevant_images")
MIN_QUOTE_CHARS = 12

ConfidenceLevel = Literal["high", "medium", "low"]


def _norm(text: object) -> str:
    return " ".join(str(text).lower().split())


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


# ── Citations ───────────────────────────────────────────────────────────────


@dataclass
class CitationReport:
    kept: dict[str, list[str]] = field(default_factory=dict)
    dropped: dict[str, list[str]] = field(default_factory=dict)

    @property
    def cited(self) -> int:
        return sum(len(v) for v in self.kept.values()) + self.fabricated

    @property
    def fabricated(self) -> int:
        return sum(len(v) for v in self.dropped.values())

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "cited": self.cited, "fabricated": self.fabricated}


def _is_present(entry: str, ids: set[str], texts: Sequence[str]) -> bool:
    """An entry is present if it is a shown id, or a verbatim quote of shown text."""
    bare = entry.strip().strip("[]").strip()
    if bare in ids:
        return True
    quote = _norm(bare)
    return len(quote) >= MIN_QUOTE_CHARS and any(quote in t for t in texts)


def validate_citations(answer: dict[str, Any], context: PackedContext) -> CitationReport:
    """Split each citation field into entries present in ``context`` and fabricated ones."""
    sources: dict[str, tuple[set[str], list[str]]] = {
        "source_chunks_used": (
            set(context.chunk_ids),
            [_norm(getattr(c, "text", c)) for c in context.text_chunks],
        ),
        "graph_paths_used": (
            {graph_path_id(i) for i in range(len(context.kept_graph_paths))},
            [_norm(p) for p in context.kept_graph_paths],
        ),
        "relevant_images": (
            set(context.image_ids),
            [
                _norm(f"{getattr(i, 'caption', '')} {getattr(i, 'image_path', '')}")
                for i in context.images
            ],
        ),
    }
    report = CitationReport()
    for name in CITATION_FIELDS:
        ids, texts = sources[name]
        entries = _as_list(answer.get(name))
        report.kept[name] = [e for e in entries if _is_present(e, ids, texts)]
        report.dropped[name] = [e for e in entries if not _is_present(e, ids, texts)]
    return report


# ── Evidence-based confidence ───────────────────────────────────────────────

HIGH_THRESHOLD = 0.60
MEDIUM_THRESHOLD = 0.35


@dataclass
class EvidenceConfidence:
    score: float
    level: ConfidenceLevel
    components: dict[str, float]


def evidence_confidence(
    text_hits: Sequence[Any],
    graph_paths_in_context: int | None,
) -> EvidenceConfidence:
    """Average of up to three components, each in [0, 1].

    * ``agreement``: share of retrieved chunks returned by both dense and BM25.
    * ``margin``: relative score drop from the first to the last retrieved
      chunk, doubled and capped at 1. A clear winner scores high.
    * ``graph``: 1 if at least one graph path made it into the prompt. Omitted
      when the graph layer is disabled (``graph_paths_in_context is None``).

    The weights and the thresholds (0.60 high, 0.35 medium) were fixed before
    any evaluation; the calibration report checks whether they track recall.
    """
    components: dict[str, float] = {}
    if text_hits:
        components["agreement"] = sum(
            1.0 for h in text_hits if len(getattr(h, "retrieved_by", ())) == 2
        ) / len(text_hits)
        top, last = float(text_hits[0].score), float(text_hits[-1].score)
        components["margin"] = min(1.0, 2 * (top - last) / top) if top > 0 else 0.0
    else:
        components["agreement"] = components["margin"] = 0.0
    if graph_paths_in_context is not None:
        components["graph"] = 1.0 if graph_paths_in_context > 0 else 0.0

    score = round(sum(components.values()) / len(components), 4)
    level: ConfidenceLevel = (
        "high" if score >= HIGH_THRESHOLD else "medium" if score >= MEDIUM_THRESHOLD else "low"
    )
    return EvidenceConfidence(score, level, components)
