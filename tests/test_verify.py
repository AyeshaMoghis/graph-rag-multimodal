"""Citation validation, evidence-based confidence, and the answer-mode summary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from graphrag.eval.calibration import calibration_table, plot_calibration
from graphrag.eval.harness import summarize
from graphrag.reasoning.answer import generate_answer, render_prompt
from graphrag.reasoning.verify import evidence_confidence, validate_citations
from graphrag.retrieval.fusion import BudgetConfig, pack_context
from graphrag.retrieval.hybrid import FusedHit


@dataclass
class Img:
    image_id: str
    caption: str
    image_path: str


def hit(cid: str, score: float, both: bool = True, text: str | None = None) -> FusedHit:
    return FusedHit(
        chunk_id=cid,
        text=text or f"{cid} explains user credibility on Twitter in detail",
        doc_id="lec.pdf",
        page=3,
        page_type="slide",
        score=score,
        dense_rank=1,
        sparse_rank=1 if both else None,
    )


def context(paths: list[str] | None = None):  # type: ignore[no-untyped-def]
    return pack_context(
        paths or [],
        [],
        [hit("c1", 0.03), hit("c2", 0.02)],
        [Img("lec.pdf_p004_img00", "a bar chart of credibility types", "out/p004_img00.png")],
        BudgetConfig(),
    )


# ── Citation validation ─────────────────────────────────────────────────────


def test_graph_citation_with_no_paths_in_context_is_fabricated() -> None:
    """The logged failure: graph_paths_used was non-empty while zero paths were in the prompt."""
    report = validate_citations({"graph_paths_used": ["MBTI -[IS_A]-> Framework"]}, context())
    assert report.fabricated == 1
    assert report.kept["graph_paths_used"] == []


def test_valid_ids_and_quotes_are_kept() -> None:
    ctx = context(["MBTI -[IS_A]-> Personality Model"])
    answer = {
        "source_chunks_used": ["c1", "[c2]", "explains user credibility on twitter", "c99"],
        "graph_paths_used": ["G1", "G2", "MBTI -[IS_A]-> Personality Model"],
        "relevant_images": ["lec.pdf_p004_img00", "img_42"],
    }
    report = validate_citations(answer, ctx)
    assert report.kept["source_chunks_used"] == [
        "c1",
        "[c2]",
        "explains user credibility on twitter",
    ]
    assert report.dropped == {
        "source_chunks_used": ["c99"],
        "graph_paths_used": ["G2"],
        "relevant_images": ["img_42"],
    }
    assert (report.cited, report.fabricated) == (9, 3)


def test_short_fragments_do_not_count_as_quotes() -> None:
    report = validate_citations({"source_chunks_used": ["user"]}, context())
    assert report.fabricated == 1


def test_truncated_graph_paths_cannot_be_cited() -> None:
    paths = [f"Node{i} -[USES]-> Other{i}" for i in range(400)]
    ctx = pack_context(paths, [], [], [], BudgetConfig(total_tokens=300))
    assert 0 < len(ctx.kept_graph_paths) < len(paths)
    report = validate_citations({"graph_paths_used": [paths[-1]]}, ctx)
    assert report.fabricated == 1


# ── Prompts and answer post-processing ──────────────────────────────────────


def test_v2_prompt_labels_every_item_and_v1_does_not() -> None:
    ctx = context(["MBTI -[IS_A]-> Personality Model"])
    v2 = render_prompt("q?", ctx, "v2")
    assert "[c1] (lec.pdf, page 3)" in v2 and "[G1] MBTI" in v2 and "[lec.pdf_p004_img00]" in v2
    v1 = render_prompt("q?", ctx, "v1")
    assert "[c1]" not in v1 and "[G1]" not in v1


class FakeLLM:
    def __init__(self, reply: dict[str, object]) -> None:
        self.reply = json.dumps(reply)

    def complete(self, prompt: str) -> str:
        return self.reply


def test_generate_answer_drops_fabricated_citations_and_keeps_self_report() -> None:
    llm = FakeLLM(
        {
            "answer": "x",
            "source_chunks_used": ["c1", "c404"],
            "graph_paths_used": ["G1"],
            "relevant_images": [],
            "confidence": "high",
        }
    )
    out = generate_answer("q?", context(), llm)
    assert out["source_chunks_used"] == ["c1"]
    assert out["graph_paths_used"] == []
    assert out["_citations"]["fabricated"] == 2
    assert out["confidence_self_reported"] == "high" and "confidence" not in out


# ── Evidence-based confidence ───────────────────────────────────────────────


def test_evidence_confidence_levels() -> None:
    strong = [hit("a", 0.033), hit("b", 0.02), hit("c", 0.016)]
    weak = [hit("a", 0.016, both=False), hit("b", 0.0159, both=False)]
    assert evidence_confidence(strong, graph_paths_in_context=4).level == "high"
    assert evidence_confidence(weak, graph_paths_in_context=0).level == "low"
    assert evidence_confidence([], graph_paths_in_context=None).score == 0.0


def test_evidence_confidence_omits_graph_when_disabled() -> None:
    hits = [hit("a", 0.033), hit("b", 0.02)]
    assert "graph" not in evidence_confidence(hits, None).components
    assert evidence_confidence(hits, 0).score < evidence_confidence(hits, None).score


# ── Summary and calibration ─────────────────────────────────────────────────


def row(i: int, recall: float, conf: str, self_conf: str, cited: int, fab: int) -> dict:
    return {
        "id": f"q{i}",
        "question_type": "factual",
        "keyword_recall": recall,
        "context_keyword_recall": recall,
        "latency_ms": 100.0,
        "graph_utilized": True,
        "graph_utilized_validated": fab == 0,
        "graph_paths_in_context": 0,
        "confidence": conf,
        "confidence_self_reported": self_conf,
        "citations_cited": cited,
        "citations_fabricated": fab,
    }


def test_answer_summary_reports_fabrication_and_calibration(tmp_path: Path) -> None:
    rows = [
        row(1, 0.9, "high", "high", 4, 0),
        row(2, 0.2, "low", "high", 3, 2),
        row(3, 0.5, "medium", "high", 2, 1),
    ]
    s = summarize(rows, "answer")
    assert s["fabricated_citation_rate"] == round(3 / 9, 3)
    assert s["answers_with_fabrication_rate"] == round(2 / 3, 3)
    assert s["graph_utilization_self_reported"] == 1.0
    assert s["graph_utilization_validated"] == round(1 / 3, 3)
    assert s["calibration"]["confidence_self_reported"] == {
        "high": {"n": 3, "keyword_recall": 0.533}
    }
    assert s["calibration"]["confidence"]["low"] == {"n": 1, "keyword_recall": 0.2}

    assert calibration_table(rows)["confidence"]["high"]["n"] == 1
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"results": rows}))
    plot_calibration(path, tmp_path / "cal.png")
    assert (tmp_path / "cal.png").stat().st_size > 0
