from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from graphrag.ingest.dedupe import ImageDeduplicator


def png(array: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(array.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def gradient(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(8, 8, 3))
    return np.kron(base, np.ones((16, 16, 1)))  # 128x128 blocky image


def test_exact_duplicate_detected() -> None:
    d = ImageDeduplicator()
    data = png(gradient(1))
    assert not d.check_and_register(data).is_duplicate
    result = d.check_and_register(data)
    assert result.is_duplicate and result.reason == "exact"


def test_near_duplicate_detected_by_phash() -> None:
    d = ImageDeduplicator(threshold=10)
    img = gradient(2)
    brighter = np.clip(img + 3, 0, 255)  # different bytes, same structure
    assert not d.check_and_register(png(img)).is_duplicate
    result = d.check_and_register(png(brighter))
    assert result.is_duplicate and result.reason == "perceptual"


def test_distinct_images_kept() -> None:
    d = ImageDeduplicator()
    for seed in range(5):
        assert not d.check_and_register(png(gradient(100 + seed))).is_duplicate
    assert d.registered == 5 and d.duplicates == 0


def test_undecodable_bytes_skip_perceptual_check() -> None:
    d = ImageDeduplicator()
    result = d.check_and_register(b"not an image")
    assert not result.is_duplicate and result.phash is None
    assert d.check_and_register(b"not an image").reason == "exact"
