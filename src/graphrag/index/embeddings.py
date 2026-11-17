"""BGE-large-en-v1.5 wrapper: lazy loading, batching, normalised output."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class Embedder:
    """Encodes text to unit-length vectors so cosine similarity is a dot product.

    The model loads on first use. Single-query encodings are kept in a small
    LRU cache because eval and ablation runs repeat the same questions.
    """

    def __init__(self, model_name: str, batch_size: int = 64, cache_size: int = 1024) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None
        self._cache: OrderedDict[str, NDArray[np.float32]] = OrderedDict()
        self._cache_size = cache_size

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str], show_progress: bool = False) -> NDArray[np.float32]:
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, query: str) -> NDArray[np.float32]:
        if query in self._cache:
            self._cache.move_to_end(query)
            return self._cache[query]
        vector: NDArray[np.float32] = np.asarray(self.encode([query])[0], dtype=np.float32)
        self._cache[query] = vector
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return vector
