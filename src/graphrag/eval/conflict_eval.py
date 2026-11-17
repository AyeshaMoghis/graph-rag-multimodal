"""Measure the contradiction detector two ways.

1. On the real corpus: how many conflicts each method flags in the top-5
   retrieved chunks for every golden question. Lecture slides from one course
   contain few or no genuine contradictions, so flags here are mostly false
   positives and must be read manually.
2. On a small synthetic planted set (``data/golden_qa/planted_conflicts.json``):
   recall on contradiction pairs and false-positive rate on consistent
   paraphrases. Without this, a detector that never fires would look perfect
   on (1).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graphrag.reasoning.verify import NLIScorer, detect_conflicts


@dataclass
class _Chunk:
    chunk_id: str
    text: str


def evaluate_planted(
    path: Path,
    nli: NLIScorer,
    encoder: Any,
    threshold: float,
    min_similarity: float,
) -> dict[str, Any]:
    pairs = json.loads(path.read_text(encoding="utf-8"))["pairs"]
    flagged = {"contradiction": 0, "consistent": 0}
    totals = {"contradiction": 0, "consistent": 0}
    for p in pairs:
        totals[p["label"]] += 1
        chunks = [_Chunk("original", p["original"]), _Chunk("planted", p["planted"])]
        found = detect_conflicts(
            chunks, [p["entity"]], nli, threshold, encoder=encoder, min_similarity=min_similarity
        )
        flagged[p["label"]] += int(bool(found))
    return {
        "contradiction_recall": round(flagged["contradiction"] / totals["contradiction"], 3),
        "consistent_false_positive_rate": round(flagged["consistent"] / totals["consistent"], 3),
        "n_contradiction": totals["contradiction"],
        "n_consistent": totals["consistent"],
    }
