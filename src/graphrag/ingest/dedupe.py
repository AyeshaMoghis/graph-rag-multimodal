"""Image deduplication: exact (MD5 of raw bytes) then perceptual (pHash).

Ported from the original notebook. Perceptual matching is a linear scan over
registered hashes; Task 2.4 replaces it with a BK-tree.
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
    _phashes: list[imagehash.ImageHash] = field(default_factory=list)
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
            for existing in self._phashes:
                if ph - existing < self.threshold:
                    self.duplicates += 1
                    return DedupResult(True, "perceptual", md5, str(ph))
            self._phashes.append(ph)
        return DedupResult(False, None, md5, str(ph) if ph is not None else None)

    @property
    def registered(self) -> int:
        return len(self._phashes)
