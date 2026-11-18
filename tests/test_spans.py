"""Sentence-level citations (Task 3.5), with the NLI model faked."""

from __future__ import annotations

import json
from collections.abc import Sequence

from graphrag.reasoning.answer import generate_answer, render_prompt
from graphrag.reasoning.verify import split_answer, verify_sentences
from graphrag.retrieval.fusion import BudgetConfig, pack_context
from graphrag.retrieval.hybrid import FusedHit


def chunk(cid: str, text: str) -> FusedHit:
    return FusedHit(cid, text, "lec.pdf", 7, "slide", 0.03, dense_rank=1, sparse_rank=1)


CTX = pack_context(
    ["MBTI -[IS_A]-> Personality Model"],
    [],
    [
        chunk("c1", "Credibility is a node property, whereas trust is an edge property."),
        chunk("c2", "Lurkers visit platforms without posting."),
    ],
    [],
    BudgetConfig(),
)


class WordOverlapNLI:
    """Entailment = share of hypothesis words found in the premise."""

    def entailment_probs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        out = []
        for premise, hyp in pairs:
            words = [w.strip(".,").lower() for w in hyp.split()]
            out.append(sum(w in premise.lower() for w in words) / len(words))
        return out


def test_split_answer_extracts_ids_and_cleans_text() -> None:
    parts = split_answer("Credibility is a node property [c1]. Trust is an edge property [c1, G1].")
    assert parts == [
        ("Credibility is a node property.", ["c1"]),
        ("Trust is an edge property.", ["c1", "G1"]),
    ]


def test_verify_sentences_marks_each_failure_mode() -> None:
    answer = (
        "Credibility is a node property [c1]. "
        "Lurkers post constantly and moderate forums daily [c2]. "
        "Trolls are bots [c99]. "
        "MBTI is a personality model [G1]. "
        "This sentence has no citation."
    )
    result = verify_sentences(answer, CTX, WordOverlapNLI(), threshold=0.6)
    supported = [s.supported for s in result]
    assert supported == [True, False, False, True, False]
    assert result[1].valid_ids == ["c2"] and result[1].entailment < 0.6  # cited but not entailed
    assert result[2].fabricated_ids == ["c99"]  # unknown id
    assert result[3].sources[0]["graph_path"].startswith("MBTI")
    assert result[0].sources[0] == {"id": "c1", "doc_id": "lec.pdf", "page": 7}


def test_v3_prompt_requires_per_sentence_citations() -> None:
    assert "End EVERY sentence" in render_prompt("q?", CTX, "v3")
    assert "End EVERY sentence" not in render_prompt("q?", CTX, "v2")


class FakeLLM:
    def complete(self, prompt: str) -> str:
        return json.dumps(
            {
                "answer": "Credibility is a node property [c1]. Lurkers write long essays [c2].",
                "source_chunks_used": ["c1", "c2"],
                "graph_paths_used": [],
                "relevant_images": [],
                "confidence": "high",
            }
        )


def test_generate_answer_reports_unsupported_claims() -> None:
    out = generate_answer(
        "q?", CTX, FakeLLM(), "v3", nli=WordOverlapNLI(), entailment_threshold=0.6
    )
    assert out["unsupported_claims"] == ["Lurkers write long essays."]
    assert out["supported_sentence_rate"] == 0.5
    assert out["sentences"][0]["supported"] is True
