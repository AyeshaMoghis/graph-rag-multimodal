"""On-disk cache for LLM completions and query embeddings (Task 3.6).

Keys are ``sha256(namespace, config hash, normalised text)``. The config hash
covers every setting that can change the output (model, temperature, prompt
version, ...), so a settings change never returns a stale entry.

SQLite is used because it ships with Python, survives restarts, and handles
concurrent readers; Redis would be the next step for a multi-process service.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def normalise(text: str, lower: bool = True) -> str:
    """Collapse whitespace; lower-case unless ``lower`` is False (LLM prompts keep case)."""
    collapsed = " ".join(text.split())
    return collapsed.lower() if lower else collapsed


def config_hash(values: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    tokens_saved: int = 0
    by_namespace: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, namespace: str, hit: bool, tokens: int = 0) -> None:
        ns = self.by_namespace.setdefault(namespace, {"hits": 0, "misses": 0})
        if hit:
            self.hits += 1
            ns["hits"] += 1
            self.tokens_saved += tokens
        else:
            self.misses += 1
            ns["misses"] += 1

    def as_dict(self, usd_per_million_tokens: float | None = None) -> dict[str, Any]:
        total = self.hits + self.misses
        out: dict[str, Any] = {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else None,
            "llm_tokens_saved": self.tokens_saved,
            "by_namespace": self.by_namespace,
        }
        if usd_per_million_tokens is not None:
            out["usd_saved"] = round(self.tokens_saved * usd_per_million_tokens / 1e6, 4)
        return out


class DiskCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value BLOB, tokens INTEGER)"
        )
        self._lock = threading.Lock()
        self.stats = CacheStats()

    @staticmethod
    def key(namespace: str, cfg: str, text: str, lower: bool = True) -> str:
        payload = f"{namespace}\x00{cfg}\x00{normalise(text, lower)}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> tuple[Any, int] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value, tokens FROM cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return pickle.loads(row[0]), int(row[1] or 0)

    def put(self, key: str, value: Any, tokens: int = 0) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, tokens) VALUES (?, ?, ?)",
                (key, pickle.dumps(value), tokens),
            )
            self._conn.commit()


class CachedLLM:
    """Wraps any LLM with ``complete``; caches by prompt and config."""

    def __init__(self, llm: Any, cache: DiskCache, cfg: str) -> None:
        self.llm = llm
        self.cache = cache
        self.cfg = cfg

    @property
    def last_usage(self) -> dict[str, int]:
        return dict(getattr(self.llm, "last_usage", {}) or {})

    def complete(self, prompt: str) -> str:
        key = self.cache.key("llm", self.cfg, prompt, lower=False)
        hit = self.cache.get(key)
        if hit is not None:
            self.cache.stats.record("llm", True, hit[1])
            return str(hit[0])
        out = self.llm.complete(prompt)
        usage = self.last_usage
        tokens = int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))
        self.cache.put(key, out, tokens)
        self.cache.stats.record("llm", False)
        return str(out)


class CachedEmbedder:
    """Wraps an embedder; single query encodings are cached on disk."""

    def __init__(self, embedder: Any, cache: DiskCache, cfg: str) -> None:
        self.embedder = embedder
        self.cache = cache
        self.cfg = cfg

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.asarray(self.embedder.encode(texts), dtype=np.float32)

    def encode_query(self, query: str) -> NDArray[np.float32]:
        key = self.cache.key("embed", self.cfg, query)
        hit = self.cache.get(key)
        if hit is not None:
            self.cache.stats.record("embed", True)
            return np.asarray(hit[0], dtype=np.float32)
        vec = np.asarray(self.embedder.encode_query(query), dtype=np.float32)
        self.cache.put(key, vec)
        self.cache.stats.record("embed", False)
        return vec
