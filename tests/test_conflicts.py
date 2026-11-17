"""NLI-based contradiction detection, with the NLI model faked."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from graphrag.reasoning.verify import candidate_pairs, detect_conflicts, split_claims
from graphrag.retrieval.fusion import naive_conflicts


@dataclass
class C:
    chunk_id: str
    text: str


class FakeNLI:
    """Contradiction when exactly one side contains ' not ' (symmetric)."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, str]] = []

    def contradiction_probs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self.seen.extend(pairs)
        return [0.95 if (" not " in a) != (" not " in b) else 0.05 for a, b in pairs]


class OneWayNLI:
    """Flags every pair, but only when read in one direction, like a small NLI model
    labelling unrelated statements as contradictory."""

    def contradiction_probs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [0.99 if a < b else 0.01 for a, b in pairs]


SLIDE = (
    "● The existence of trust between user nodes finally leads to the\n"
    "formation of edges that lead to community formation\n"
    "● Twitter does not provide an explicit trust score. Implicit trust score\n"
    "can be calculated based on the local behavior of the node.\n"
    "Rumor, Fake News and"
)


def test_split_claims_rejoins_wrapped_lines_and_drops_titles() -> None:
    assert split_claims(SLIDE) == [
        "The existence of trust between user nodes finally leads to the formation of edges "
        "that lead to community formation",
        "Twitter does not provide an explicit trust score.",
        "Implicit trust score can be calculated based on the local behavior of the node.",
    ]


def test_candidate_pairs_need_shared_entity_and_different_chunks() -> None:
    a = "Trolls target individual users directly and repeatedly online."
    chunks = [
        C("a", a + " Lurkers rarely post anything at all in communities."),
        C("b", "Trolls do not target individual users directly at all."),
        C("c", a),  # identical claim to chunk a
    ]
    pairs = candidate_pairs(chunks, ["trolls", "lurkers"])
    # a/c are identical so never paired; b/c would repeat the a/b judgment word for
    # word, so it is deduplicated. Repeated slides are common in this corpus.
    assert {(x[0], y[0]) for _, x, y in pairs} == {("a", "b")}
    assert all(entity == "trolls" for entity, _, _ in pairs)


def test_candidate_pairs_are_capped() -> None:
    chunks = [C(str(i), f"Trolls behave in pattern number {i} online today.") for i in range(20)]
    assert len(candidate_pairs(chunks, ["trolls"], max_pairs=10)) == 10


def test_detect_conflicts_flags_contradictions_above_threshold() -> None:
    chunks = [
        C("a", "Cyberbullies target a single individual repeatedly over time."),
        C("b", "Cyberbullies do not target a single individual over time."),
        C("c", "Cyberbullies post images, text, audio and video content."),
    ]
    nli = FakeNLI()
    found = detect_conflicts(chunks, ["cyberbullies"], nli, threshold=0.8)
    assert len(nli.seen) == 6  # three pairs, scored in both directions
    assert {(c.source_a, c.source_b) for c in found} == {("a", "b"), ("b", "c")}
    assert "Possible contradiction about 'cyberbullies'" in found[0].describe()


def test_one_directional_contradictions_are_rejected() -> None:
    chunks = [
        C("a", "Trolls post inflammatory comments in public sections."),
        C("b", "Trolls focus their effort on a small number of threads."),
    ]
    assert detect_conflicts(chunks, ["trolls"], OneWayNLI()) == []


def test_naive_check_fires_on_contrastive_phrasing_without_contradiction() -> None:
    """The original heuristic flags 'unlike trolls' even though nothing is contradicted."""
    chunks = [C("a", "Unlike trolls, cyberbullies target one individual repeatedly.")]
    assert naive_conflicts(chunks, ["trolls"]) != []
    assert detect_conflicts(chunks, ["trolls"], FakeNLI()) == []


class AxisEncoder:
    """Claims mentioning 'detector' point one way, everything else another."""

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = [[1.0, 0.0] if "detector" in t else [0.0, 1.0] for t in texts]
        return np.array(rows, dtype=np.float32)


def test_relatedness_gate_skips_unrelated_pairs() -> None:
    chunks = [
        C("a", "Twitter was used to build an earthquake detector for Australia."),
        C("b", "Twitter data did not predict the Dutch Senate elections well."),
    ]
    assert detect_conflicts(chunks, ["twitter"], FakeNLI()) != []  # ungated: flagged
    gated = detect_conflicts(
        chunks, ["twitter"], FakeNLI(), encoder=AxisEncoder(), min_similarity=0.6
    )
    assert gated == []
