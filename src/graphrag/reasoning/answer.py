"""Structured-JSON answer generation over packed context (ported)."""

from __future__ import annotations

import json
from typing import Any

from graphrag.graph.extract import strip_fences
from graphrag.reasoning.llm import LLM
from graphrag.reasoning.prompts import render_reasoning_prompt
from graphrag.retrieval.fusion import PackedContext

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


def generate_answer(query: str, context: PackedContext, llm: LLM) -> dict[str, Any]:
    prompt = render_reasoning_prompt(
        query,
        context.graph_context,
        [str(getattr(c, "text", c)) for c in context.text_chunks],
        [str(getattr(i, "caption", i)) for i in context.images],
        context.conflicts,
    )
    try:
        result = parse_answer(llm.complete(prompt))
    except Exception as exc:
        result = parse_answer(f"[LLM error: {exc}]")
    result["_fusion_metadata"] = context.metadata()
    result["_query"] = query
    return result
