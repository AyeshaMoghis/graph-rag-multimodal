"""Image deduplication: exact (MD5 of raw bytes) then perceptual (pHash).

Ported from the original notebook, with one change: the notebook compared each
new pHash against every registered hash (O(n) per image). Near-duplicate lookup
now uses a BK-tree over Hamming distance, which prunes any subtree whose edge
label is further than the search radius from the query's distance to its root.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from io import BytesIO

import imagehash
from PIL import Image


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def perceptual_hash(data: bytes) -> imagehash.ImageHash | None:
    """pHash of an image, or None if the bytes cannot be decoded."""
    try:
        with Image.open(BytesIO(data)) as img:
            return imagehash.phash(img.convert("RGB"))
    except Exception:
        return None


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class BKTree:
    """Burkhard-Keller tree for integer hashes under Hamming distance."""

    def __init__(self) -> None:
        self._root: tuple[int, dict[int, tuple[int, dict]]] | None = None  # type: ignore[type-arg]
        self._size = 0
        self.comparisons = 0  # distance computations, for benchmarking

    def __len__(self) -> int:
        return self._size

    def add(self, value: int) -> None:
        if self._root is None:
            self._root = (value, {})
            self._size = 1
            return
        node = self._root
        while True:
            d = hamming(value, node[0])
            if d == 0:
                return  # already present
            child = node[1].get(d)
            if child is None:
                node[1][d] = (value, {})
                self._size += 1
                return
            node = child

    def any_within(self, value: int, max_distance: int) -> bool:
        """True if some stored value is within ``max_distance`` (inclusive)."""
        if self._root is None:
            return False
        stack = [self._root]
        while stack:
            node_value, children = stack.pop()
            self.comparisons += 1
            d = hamming(value, node_value)
            if d <= max_distance:
                return True
            for edge, child in children.items():
                if d - max_distance <= edge <= d + max_distance:
                    stack.append(child)
        return False


@dataclass
class DedupResult:
    is_duplicate: bool
    reason: str | None  # "exact" | "perceptual" | None
    md5: str
    phash: str | None


@dataclass
class ImageDeduplicator:
    """Stateful deduplicator shared across every PDF in a corpus.

    A new image is a near-duplicate when its pHash Hamming distance to any
    registered hash is strictly below ``threshold`` (original: 10).
    """

    threshold: int = 10
    _exact: set[str] = field(default_factory=set)
    _tree: BKTree = field(default_factory=BKTree)
    checked: int = 0
    duplicates: int = 0

    def check_and_register(self, data: bytes) -> DedupResult:
        self.checked += 1
        md5 = md5_bytes(data)
        if md5 in self._exact:
            self.duplicates += 1
            return DedupResult(True, "exact", md5, None)
        self._exact.add(md5)

        ph = perceptual_hash(data)
        if ph is not None:
            value = int(str(ph), 16)
            # The original rule is "distance strictly below threshold".
            if self._tree.any_within(value, self.threshold - 1):
                self.duplicates += 1
                return DedupResult(True, "perceptual", md5, str(ph))
            self._tree.add(value)
        return DedupResult(False, None, md5, str(ph) if ph is not None else None)

    @property
    def registered(self) -> int:
        return len(self._tree)
