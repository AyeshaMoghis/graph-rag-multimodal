"""Structured-JSON answer generation over packed context (ported)."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Literal

from graphrag.graph.extract import strip_fences
from graphrag.reasoning.llm import LLM
from graphrag.reasoning.prompts import (
    REASONING_PROMPT_V2,
    REASONING_PROMPT_V3,
    render_reasoning_prompt,
    render_reasoning_prompt_v2,
)
from graphrag.reasoning.verify import EntailmentScorer, validate_citations, verify_sentences
from graphrag.retrieval.fusion import PackedContext

PromptVersion = Literal["v1", "v2", "v3"]

ANSWER_KEYS = (
    "answer",
    "referenced_concepts",
    "source_chunks_used",
    "graph_paths_used",
    "relevant_images",
    "confidence",
)


def parse_answer(raw: str) -> dict[str, Any]:
    """Parse the model's JSON. On failure keep the raw text as the answer, confidence low."""
    try:
        data = json.loads(strip_fences(raw))
        if not isinstance(data, dict):
            raise ValueError("answer is not a JSON object")
        return data
    except Exception:
        return {
            "answer": raw,
            "referenced_concepts": [],
            "source_chunks_used": [],
            "graph_paths_used": [],
            "relevant_images": [],
            "confidence": "low",
            "_parse_error": True,
        }


def render_prompt(query: str, context: PackedContext, prompt_version: PromptVersion) -> str:
    if prompt_version == "v1":
        return render_reasoning_prompt(
            query,
            context.graph_context,
            [str(getattr(c, "text", c)) for c in context.text_chunks],
            [str(getattr(i, "caption", i)) for i in context.images],
            context.conflicts,
        )
    return render_reasoning_prompt_v2(
        query,
        context.kept_graph_paths,
        [
            (
                str(getattr(c, "chunk_id", "")),
                str(getattr(c, "doc_id", "")),
                int(getattr(c, "page", 0)),
                str(getattr(c, "text", c)),
            )
            for c in context.text_chunks
        ],
        [(str(getattr(i, "image_id", "")), str(getattr(i, "caption", i))) for i in context.images],
        context.conflicts,
        template=REASONING_PROMPT_V3 if prompt_version == "v3" else REASONING_PROMPT_V2,
    )


def generate_answer(
    query: str,
    context: PackedContext,
    llm: LLM,
    prompt_version: PromptVersion = "v3",
    nli: EntailmentScorer | None = None,
    entailment_threshold: float = 0.5,
) -> dict[str, Any]:
    """Generate, then validate citations: entries not present in the prompt are removed.

    With an NLI model, every answer sentence is also checked against the sources
    it cites; sentences that are uncited, cite an unknown id, or are not entailed
    by any cited source are listed in ``unsupported_claims``.
    """
    try:
        result = parse_answer(llm.complete(render_prompt(query, context, prompt_version)))
    except Exception as exc:
        result = parse_answer(f"[LLM error: {exc}]")
        result["_llm_error"] = str(exc)  # callers must not score this as an answer
    report = validate_citations(result, context)
    for name, kept in report.kept.items():
        result[name] = kept
    result["_citations"] = report.to_dict()
    if nli is not None and isinstance(result.get("answer"), str) and not result.get("_parse_error"):
        sentences = verify_sentences(result["answer"], context, nli, entailment_threshold)
        result["sentences"] = [asdict(s) for s in sentences]
        result["unsupported_claims"] = [s.text for s in sentences if not s.supported]
        result["supported_sentence_rate"] = (
            round(sum(s.supported for s in sentences) / len(sentences), 3) if sentences else 0.0
        )
    result["confidence_self_reported"] = result.pop("confidence", None)
    result["_fusion_metadata"] = context.metadata()
    result["_query"] = query
    result["_prompt_version"] = prompt_version
    return result
