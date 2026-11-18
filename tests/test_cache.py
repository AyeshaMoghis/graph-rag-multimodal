from __future__ import annotations

from pathlib import Path

import numpy as np

from graphrag.cache import CachedEmbedder, CachedLLM, DiskCache, config_hash


class CountingLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.last_usage = {"prompt_tokens": 90, "completion_tokens": 10}

    def complete(self, prompt: str) -> str:
        self.calls += 1
        return f"answer to {prompt}"


def test_llm_cache_hits_survive_restarts_and_count_tokens(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    llm = CountingLLM()
    cached = CachedLLM(llm, DiskCache(path), config_hash({"model": "m"}))
    assert cached.complete("Q  one") == cached.complete("Q one")  # whitespace normalised
    assert llm.calls == 1
    reopened = CachedLLM(CountingLLM(), DiskCache(path), config_hash({"model": "m"}))
    reopened.complete("Q one")
    assert reopened.llm.calls == 0  # served from disk
    stats = reopened.cache.stats.as_dict(usd_per_million_tokens=0.05)
    assert stats["hits"] == 1 and stats["llm_tokens_saved"] == 100
    assert stats["usd_saved"] == round(100 * 0.05 / 1e6, 4)


def test_llm_cache_is_case_sensitive_and_config_scoped(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "c.sqlite")
    llm = CountingLLM()
    CachedLLM(llm, cache, "cfgA").complete("MBTI")
    CachedLLM(llm, cache, "cfgA").complete("mbti")
    CachedLLM(llm, cache, "cfgB").complete("MBTI")
    assert llm.calls == 3


class CountingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        self.calls += 1
        return np.array([1.0, 0.0], dtype=np.float32)


def test_query_embedding_cache(tmp_path: Path) -> None:
    emb = CountingEmbedder()
    cached = CachedEmbedder(emb, DiskCache(tmp_path / "c.sqlite"), "bge")
    cached.encode_query("What is MBTI?")
    out = cached.encode_query("what is  MBTI?")
    assert emb.calls == 1 and out.tolist() == [1.0, 0.0]
    assert cached.cache.stats.as_dict()["hit_rate"] == 0.5
