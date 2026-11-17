from __future__ import annotations

from pathlib import Path

from graphrag.index.sparse import BM25Index, legacy_query_tokenize, normalized_tokenize

CORPUS = {
    "c1": "MBTI defines sixteen personality types.",
    "c2": "Cyberbullying harms mental health, especially for teens.",
    "c3": "Graph neural networks, unlike CNNs, operate on graphs.",
    "c4": "Credibility of users on social platforms varies.",
}


def make(tokenizer: str) -> BM25Index:
    return BM25Index(list(CORPUS), list(CORPUS.values()), tokenizer=tokenizer)  # type: ignore[arg-type]


def test_normalized_tokenizer_strips_case_and_punctuation() -> None:
    assert normalized_tokenize("MBTI, Networks!") == ["mbti", "networks"]


def test_legacy_tokenizer_cannot_match_capitalised_terms() -> None:
    """Documents the original bug: lower-cased query vs case-preserved documents."""
    assert legacy_query_tokenize("MBTI") == ["mbti"]
    assert make("legacy").scores("MBTI").max() == 0.0
    assert make("normalized").search("MBTI", 1)[0].chunk_id == "c1"


def test_legacy_tokenizer_misses_words_with_trailing_punctuation() -> None:
    # "networks," in the document is a different token from "networks".
    assert make("legacy").scores("networks").max() == 0.0
    assert make("normalized").search("networks", 1)[0].chunk_id == "c3"


def test_search_returns_requested_count_in_score_order() -> None:
    hits = make("normalized").search("mental health teens", 3)
    assert len(hits) == 3
    assert hits[0].chunk_id == "c2"
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_round_trip_persistence(tmp_path: Path) -> None:
    path = tmp_path / "bm25.json"
    original = make("normalized")
    original.save(path)
    loaded = BM25Index.load(path)
    assert loaded.chunk_ids == original.chunk_ids
    assert loaded.search("credibility", 1)[0].chunk_id == "c4"
    assert BM25Index.load(path, tokenizer="legacy").tokenizer_mode == "legacy"
