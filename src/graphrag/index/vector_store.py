"""ChromaDB collections for text chunks and image captions (cosine space).

Every record stores its ``chunk_id`` in metadata as well as using it as the
Chroma id. The original notebook omitted it from metadata, which is the root of
the fusion bug documented in docs/ENGINEERING_NOTES.md.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from graphrag.index.types import ScoredHit

TEXT_COLLECTION = "text_chunks"
IMAGE_COLLECTION = "image_chunks"
_BATCH = 256


def cosine_distance_to_similarity(distance: float) -> float:
    """Map Chroma's cosine distance (0..2) onto 0..1, as the original did."""
    return 1.0 - distance / 2.0


class VectorStore:
    def __init__(self, path: Path) -> None:
        import chromadb

        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))

    def rebuild(
        self,
        name: str,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: NDArray[np.float32],
        metadatas: Sequence[dict[str, Any]],
    ) -> int:
        """Drop and recreate a collection. ``chunk_id`` is added to every metadata dict."""
        if not (len(ids) == len(documents) == len(embeddings) == len(metadatas)):
            raise ValueError("ids, documents, embeddings and metadatas must align")
        with contextlib.suppress(Exception):  # collection did not exist yet
            self._client.delete_collection(name)
        collection = self._client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
        for start in range(0, len(ids), _BATCH):
            end = start + _BATCH
            collection.add(
                ids=list(ids[start:end]),
                embeddings=embeddings[start:end].tolist(),
                documents=list(documents[start:end]),
                metadatas=[
                    {**meta, "chunk_id": cid}
                    for cid, meta in zip(ids[start:end], metadatas[start:end], strict=True)
                ],
            )
        return int(collection.count())

    def count(self, name: str) -> int:
        return int(self._client.get_collection(name).count())

    def query(self, name: str, embedding: NDArray[np.float32], n: int) -> list[ScoredHit]:
        collection = self._client.get_collection(name)
        n = min(n, collection.count())
        if n <= 0:
            return []
        res = collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        ids = res["ids"][0]
        docs = (res["documents"] or [[]])[0]
        metas = (res["metadatas"] or [[]])[0]
        dists = (res["distances"] or [[]])[0]
        return [
            ScoredHit(
                chunk_id=cid,
                text=doc or "",
                score=cosine_distance_to_similarity(dist),
                metadata=dict(meta or {}),
            )
            for cid, doc, meta, dist in zip(ids, docs, metas, dists, strict=True)
        ]


class DenseSearcher:
    """Adapts a collection plus an embedder to the ``Searcher`` protocol."""

    def __init__(self, store: VectorStore, collection: str, embedder: Any) -> None:
        self.store = store
        self.collection = collection
        self.embedder = embedder

    def search(self, query: str, n: int) -> list[ScoredHit]:
        return self.store.query(self.collection, self.embedder.encode_query(query), n)
