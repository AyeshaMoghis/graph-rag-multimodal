"""BM25Okapi sparse index with JSON persistence.

Two tokenizers are available:

* ``legacy`` reproduces the original notebook: documents were split on
  whitespace with case preserved, but queries were lower-cased. A query term can
  therefore never match a capitalised document token ("mbti" vs "MBTI"), and
  punctuation stays glued to words ("networks," vs "networks").
* ``normalized`` lower-cases and splits on word characters for both sides.

The index persists its tokenised corpus, not a pickle, and rebuilds BM25 on
load. That is fast at this corpus size and avoids unpickling untrusted files.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from graphrag.config import TokenizerMode
from graphrag.index.types import ScoredHit

_WORD = re.compile(r"\w+")

Tokenizer = Callable[[str], list[str]]


def normalized_tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def legacy_doc_tokenize(text: str) -> list[str]:
    return text.split()


def legacy_query_tokenize(text: str) -> list[str]:
    return text.lower().split()


def tokenizers_for(mode: TokenizerMode) -> tuple[Tokenizer, Tokenizer]:
    """Return (document_tokenizer, query_tokenizer)."""
    if mode == "legacy":
        return legacy_doc_tokenize, legacy_query_tokenize
    return normalized_tokenize, normalized_tokenize


class BM25Index:
    def __init__(
        self,
        chunk_ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Sequence[dict[str, object]] | None = None,
        tokenizer: TokenizerMode = "normalized",
    ) -> None:
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must align")
        self.chunk_ids = list(chunk_ids)
        self.texts = list(texts)
        self.metadatas = list(metadatas) if metadatas is not None else [{} for _ in texts]
        self.tokenizer_mode: TokenizerMode = tokenizer
        self._doc_tok, self._query_tok = tokenizers_for(tokenizer)
        self._bm25 = BM25Okapi([self._doc_tok(t) for t in self.texts])

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self._bm25.get_scores(self._query_tok(query)), dtype=np.float64)

    def search(self, query: str, n: int) -> list[ScoredHit]:
        """Top-n chunks by raw BM25 score. Ties break on corpus order for determinism."""
        raw = self.scores(query)
        order = np.argsort(-raw, kind="stable")[:n]
        return [
            ScoredHit(self.chunk_ids[i], self.texts[i], float(raw[i]), dict(self.metadatas[i]))
            for i in order
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tokenizer": self.tokenizer_mode,
            "chunk_ids": self.chunk_ids,
            "texts": self.texts,
            "metadatas": self.metadatas,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, tokenizer: TokenizerMode | None = None) -> BM25Index:
        """Load a saved corpus, optionally re-tokenising with a different mode."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            payload["chunk_ids"],
            payload["texts"],
            payload["metadatas"],
            tokenizer=tokenizer or payload["tokenizer"],
        )
