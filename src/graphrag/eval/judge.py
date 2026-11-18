"""LLM-as-judge for faithfulness and answer relevance (Task 3.3).

The judge must be a different model from the one that writes answers, so the
judge never grades its own output. It is disabled until ``GRAPHRAG_JUDGE_MODEL``
is set, and refuses to run if it equals ``GRAPHRAG_LLM_MODEL``.

Agreement with humans is measured on a small hand-labelled subset
(``data/golden_qa/judge_human_labels.json``); the file ships as a template with
no labels, because labels must come from a person.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graphrag.graph.extract import strip_fences
from graphrag.reasoning.llm import LLM

JUDGE_PROMPT = """You are grading an answer produced by a retrieval-augmented system.

CONTEXT (everything the system was given):
REPLACE_CONTEXT

QUESTION: REPLACE_QUESTION

ANSWER: REPLACE_ANSWER

Grade two things.
1. faithfulness: split the answer into its individual factual claims. A claim is
   supported only if the CONTEXT states or directly implies it; background
   knowledge does not count. faithfulness = supported claims / total claims.
2. answer_relevance: 1 to 5, how directly the answer addresses the question
   (5 = fully answers it, 1 = off-topic). Judge relevance, not correctness.

Respond ONLY with JSON:
{"claims": [{"claim": "...", "supported": true}], "faithfulness": 0.0, "answer_relevance": 1}
"""


@dataclass
class JudgeResult:
    faithfulness: float
    answer_relevance: int
    claims: list[dict[str, Any]]
    error: str | None = None


def judge_answer(question: str, context: str, answer: str, judge: LLM) -> JudgeResult:
    prompt = (
        JUDGE_PROMPT.replace("REPLACE_CONTEXT", context)
        .replace("REPLACE_QUESTION", question)
        .replace("REPLACE_ANSWER", answer)
    )
    try:
        data = json.loads(strip_fences(judge.complete(prompt)))
        claims = list(data.get("claims", []))
        # Recompute from the claim list rather than trusting the judge's arithmetic.
        faith = (
            sum(1 for c in claims if c.get("supported") is True) / len(claims)
            if claims
            else float(data.get("faithfulness", 0.0))
        )
        relevance = int(data.get("answer_relevance", 0))
        if not 1 <= relevance <= 5:
            raise ValueError(f"answer_relevance out of range: {relevance}")
        return JudgeResult(round(faith, 3), relevance, claims)
    except Exception as exc:
        return JudgeResult(0.0, 0, [], error=str(exc))


def check_judge_model(judge_model: str | None, answer_model: str) -> str:
    if not judge_model:
        raise ValueError("set GRAPHRAG_JUDGE_MODEL to a model different from the answer model")
    if judge_model == answer_model:
        raise ValueError("the judge must be a different model from the one that wrote the answers")
    return judge_model


def agreement(human: list[float], judge: list[float], threshold: float = 0.5) -> dict[str, Any]:
    """Agreement between human and judge faithfulness scores on the same answers.

    Reports binary agreement and Cohen's kappa at ``threshold`` (faithful vs
    not), and mean absolute difference of the raw scores.
    """
    if len(human) != len(judge) or not human:
        raise ValueError("need the same, non-zero number of human and judge scores")
    h = [x >= threshold for x in human]
    j = [x >= threshold for x in judge]
    n = len(h)
    observed = sum(a == b for a, b in zip(h, j, strict=True)) / n
    p_h, p_j = sum(h) / n, sum(j) / n
    expected = p_h * p_j + (1 - p_h) * (1 - p_j)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return {
        "n": n,
        "binary_agreement": round(observed, 3),
        "cohens_kappa": round(kappa, 3),
        "mean_abs_diff": round(
            statistics.fmean(abs(a - b) for a, b in zip(human, judge, strict=True)), 3
        ),
    }


def write_label_template(path: Path, rows: list[dict[str, Any]], n: int = 12) -> None:
    """Blank template for human faithfulness labels, from an answer-mode results file."""
    picked = [r for r in rows if "error" not in r][:n]
    path.write_text(
        json.dumps(
            {
                "instructions": (
                    "For each answer, read the retrieved context and set human_faithfulness "
                    "to the share of the answer's claims the context supports (0.0 to 1.0). "
                    "Leave null until labelled."
                ),
                "items": [
                    {"id": r["id"], "answer": r.get("answer", ""), "human_faithfulness": None}
                    for r in picked
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
