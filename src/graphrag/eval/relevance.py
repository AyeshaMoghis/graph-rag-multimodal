"""Chunk-level relevance labels for retrieval metrics (Task 3.3).

The golden set has expected concepts but no relevant chunks. Candidates are
generated automatically and written to a file for manual review; until a
question is reviewed, its labels are *suggestions* and every metric computed
from them is reported as provisional.

How candidates are generated, chosen to avoid circularity:

* Chunks are ranked by cosine similarity between the question's
  ``ideal_answer`` and the chunk text, using an embedding model that no
  retriever under evaluation uses (``all-MiniLM-L6-v2``). Ranking by the
  retrievers themselves or by expected-concept overlap would bias the labels
  towards the systems or the keyword metric being evaluated.
* Byte-identical chunks (the corpus repeats slides across PDFs) form one
  relevance group, so retrieving any copy counts once.
* A group is *suggested* relevant if it is among the top 3 groups, its
  similarity is at least 0.5, and it mentions at least one expected concept.
  Concept overlap is shown to help the reviewer but never ranks candidates.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

LABEL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CANDIDATES_PER_QUESTION = 10
SUGGEST_TOP_GROUPS = 3
SUGGEST_MIN_SIMILARITY = 0.5


def text_group(text: str) -> str:
    """Identical text (after whitespace normalisation) shares a group id."""
    return hashlib.sha1(" ".join(text.split()).encode("utf-8")).hexdigest()[:12]


def generate_candidates(
    questions: Sequence[dict[str, Any]],
    chunks: Sequence[Any],
    encoder: Any,
) -> dict[str, Any]:
    groups: dict[str, list[Any]] = {}
    for c in chunks:
        if c.text.strip():
            groups.setdefault(text_group(c.text), []).append(c)
    group_ids = list(groups)
    group_vecs = np.asarray(encoder.encode([groups[g][0].text for g in group_ids]))
    answer_vecs = np.asarray(encoder.encode([q["ideal_answer"] for q in questions]))

    out = []
    for q, qvec in zip(questions, answer_vecs, strict=True):
        sims = group_vecs @ qvec
        order = np.argsort(-sims, kind="stable")[:CANDIDATES_PER_QUESTION]
        candidates = []
        for rank, gi in enumerate(order):
            g = group_ids[int(gi)]
            text = groups[g][0].text
            matched = [c for c in q["expected_concepts"] if c.lower() in text.lower()]
            sim = round(float(sims[gi]), 4)
            candidates.append(
                {
                    "group": g,
                    "chunk_ids": [c.chunk_id for c in groups[g]],
                    "similarity_to_ideal_answer": sim,
                    "matched_concepts": matched,
                    "suggested": rank < SUGGEST_TOP_GROUPS
                    and sim >= SUGGEST_MIN_SIMILARITY
                    and bool(matched),
                    "relevant": None,  # set to true/false during manual review
                    "snippet": " ".join(text.split())[:300],
                }
            )
        out.append(
            {
                "id": q["id"],
                "question": q["question"],
                "question_type": q.get("question_type", ""),
                "ideal_answer": q["ideal_answer"],
                "reviewed": False,
                "candidates": candidates,
            }
        )
    return {
        "status": "UNREVIEWED: automatic suggestions, not ground truth",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": {
            "label_model": LABEL_MODEL,
            "ranking": "cosine(ideal_answer, chunk), duplicate texts grouped",
            "candidates_per_question": CANDIDATES_PER_QUESTION,
            "suggested_if": (
                f"top {SUGGEST_TOP_GROUPS} groups AND similarity >= {SUGGEST_MIN_SIMILARITY} "
                "AND at least one expected concept appears in the chunk"
            ),
            "how_to_review": (
                "For each candidate set 'relevant' to true or false, then set the "
                "question's 'reviewed' to true. Reviewed questions use 'relevant'; "
                "unreviewed ones fall back to 'suggested' and are reported as provisional."
            ),
        },
        "questions": out,
    }


def load_relevance(path: Path) -> tuple[dict[str, dict[str, list[str]]], dict[str, bool]]:
    """Return ({question_id: {group: chunk_ids}} for relevant groups, {question_id: reviewed})."""
    data = json.loads(path.read_text(encoding="utf-8"))
    labels: dict[str, dict[str, list[str]]] = {}
    reviewed: dict[str, bool] = {}
    for q in data["questions"]:
        is_reviewed = bool(q.get("reviewed"))
        reviewed[q["id"]] = is_reviewed
        labels[q["id"]] = {
            c["group"]: c["chunk_ids"]
            for c in q["candidates"]
            if (c.get("relevant") is True if is_reviewed else c.get("suggested"))
        }
    return labels, reviewed
