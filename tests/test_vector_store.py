"""VectorStore against a real on-disk Chroma instance (local only, no network)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from graphrag.index.vector_store import VectorStore


def unit(v: list[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def test_chunk_id_is_written_to_metadata(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "chroma")
    store.rebuild(
        "test_col",
        ["a", "b"],
        ["text a", "text b"],
        np.stack([unit([1, 0, 0]), unit([0, 1, 0])]),
        [{"page": 1}, {"page": 2}],
    )
    hits = store.query("test_col", unit([1, 0, 0]), 2)
    assert hits[0].chunk_id == "a"
    assert hits[0].metadata["chunk_id"] == "a"  # the field the original notebook never stored


def test_exact_ties_at_the_cutoff_resolve_by_chunk_id(tmp_path: Path) -> None:
    """Identical slides embed identically. The cut-off must not depend on HNSW order."""
    store = VectorStore(tmp_path / "chroma")
    dup = unit([1, 1, 0])
    ids = ["dup_c", "dup_a", "dup_b", "other"]
    vecs = np.stack([dup, dup, dup, unit([0, 0, 1])])
    store.rebuild("test_col", ids, ["same slide"] * 3 + ["different"], vecs, [{}] * 4)
    for _ in range(5):
        hits = store.query("test_col", dup, 2)
        assert [h.chunk_id for h in hits] == ["dup_a", "dup_b"]
