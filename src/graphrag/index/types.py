"""Types shared by every retriever."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ScoredHit:
    """One result from a single retriever, in that retriever's own score scale."""

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class TextEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]: ...


class Searcher(Protocol):
    def search(self, query: str, n: int) -> list[ScoredHit]: ...
