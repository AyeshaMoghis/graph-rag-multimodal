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


def test_bktree_matches_linear_scan_on_random_hashes() -> None:
    import random

    from graphrag.ingest.dedupe import BKTree, hamming

    rng = random.Random(0)
    stored = [rng.getrandbits(64) for _ in range(500)]
    tree = BKTree()
    for v in stored:
        tree.add(v)
    for _ in range(300):
        q = (
            rng.choice(stored) ^ (1 << rng.randrange(64))
            if rng.random() < 0.5
            else rng.getrandbits(64)
        )
        for radius in (0, 5, 9, 20):
            expected = any(hamming(q, s) <= radius for s in stored)
            assert tree.any_within(q, radius) == expected


def test_bktree_prunes_comparisons() -> None:
    import random

    from graphrag.ingest.dedupe import BKTree

    rng = random.Random(1)
    tree = BKTree()
    for _ in range(2000):
        tree.add(rng.getrandbits(64))
    tree.comparisons = 0
    for _ in range(100):
        tree.any_within(rng.getrandbits(64), 9)
    assert tree.comparisons / 100 < len(tree)  # fewer than a linear scan per query
