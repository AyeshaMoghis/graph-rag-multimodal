from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphrag.eval.judge import agreement, check_judge_model, judge_answer, write_label_template


class FakeJudge:
    def __init__(self, reply: dict) -> None:
        self.reply = json.dumps(reply)
        self.prompt = ""

    def complete(self, prompt: str) -> str:
        self.prompt = prompt
        return self.reply


def test_faithfulness_is_recomputed_from_claims() -> None:
    judge = FakeJudge(
        {
            "claims": [{"claim": "a", "supported": True}, {"claim": "b", "supported": False}],
            "faithfulness": 0.9,  # wrong arithmetic from the judge is ignored
            "answer_relevance": 4,
        }
    )
    result = judge_answer("q?", "ctx", "ans", judge)
    assert (result.faithfulness, result.answer_relevance, result.error) == (0.5, 4, None)
    assert "ctx" in judge.prompt and "ans" in judge.prompt


def test_bad_judge_output_is_an_error_not_a_score() -> None:
    assert judge_answer("q", "c", "a", FakeJudge({"claims": [], "answer_relevance": 9})).error


def test_judge_must_differ_from_answer_model() -> None:
    with pytest.raises(ValueError):
        check_judge_model(None, "llama")
    with pytest.raises(ValueError):
        check_judge_model("llama", "llama")
    assert check_judge_model("other", "llama") == "other"


def test_agreement() -> None:
    out = agreement([1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    assert out["binary_agreement"] == 0.75
    assert out["cohens_kappa"] == 0.5
    assert out["mean_abs_diff"] == 0.25


def test_label_template_has_no_labels(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    write_label_template(path, [{"id": "q1", "answer": "x"}, {"id": "q2", "error": "boom"}])
    items = json.loads(path.read_text())["items"]
    assert items == [{"id": "q1", "answer": "x", "human_faithfulness": None}]
