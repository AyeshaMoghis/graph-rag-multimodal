"""Token-budget context packing across graph, text and image layers.

Ported from the notebook's ``fuse_context``: a 6000-token budget split
40/40/20, a whitespace-based token estimate, and greedy in-order packing that
stops at the first item that would overflow its layer's budget.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


def estimate_tokens(text: str) -> int:
    """Rough token count: words / 0.75."""
    return int(len(text.split()) / 0.75)


def truncate_to_budget(items: Sequence[Any], budget_tokens: int, key: str = "text") -> list[Any]:
    """Keep items in order until the next one would exceed the budget."""
    used, kept = 0, []
    for item in items:
        text = item if isinstance(item, str) else _get(item, key)
        tokens = estimate_tokens(text)
        if used + tokens > budget_tokens:
            break
        kept.append(item)
        used += tokens
    return kept


def _get(item: Any, key: str) -> str:
    if isinstance(item, Mapping):
        return str(item.get(key, ""))
    return str(getattr(item, key, ""))


@dataclass(frozen=True)
class BudgetConfig:
    total_tokens: int = 6000
    graph_fraction: float = 0.40
    text_fraction: float = 0.40
    image_fraction: float = 0.20

    def __post_init__(self) -> None:
        total = self.graph_fraction + self.text_fraction + self.image_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"budget fractions must sum to 1.0, got {total}")


@dataclass
class PackedContext:
    graph_context: str
    graph_paths: list[str]
    text_chunks: list[Any]
    images: list[Any]
    conflicts: list[str] = field(default_factory=list)

    @property
    def chunk_ids(self) -> list[str]:
        return [_get(c, "chunk_id") for c in self.text_chunks]

    def metadata(self) -> dict[str, Any]:
        return {
            "graph_paths_total": len(self.graph_paths),
            "text_chunks_used": len(self.text_chunks),
            "images_used": len(self.images),
            "chunk_ids": self.chunk_ids,
        }


NO_GRAPH = "(no graph paths found)"


def pack_context(
    graph_paths: Sequence[str],
    graph_nodes: Sequence[str],
    text_chunks: Sequence[Any],
    images: Sequence[Any],
    budget: BudgetConfig | None = None,
) -> PackedContext:
    cfg = budget or BudgetConfig()
    graph_budget = int(cfg.total_tokens * cfg.graph_fraction)
    text_budget = int(cfg.total_tokens * cfg.text_fraction)
    image_budget = int(cfg.total_tokens * cfg.image_fraction)

    graph_str = "\n".join(graph_paths) if graph_paths else NO_GRAPH
    if estimate_tokens(graph_str) > graph_budget:
        words = graph_str.split()
        graph_str = " ".join(words[: int(graph_budget * 0.75)]) + " [truncated]"

    kept_chunks = truncate_to_budget(text_chunks, text_budget, key="text")
    kept_images = truncate_to_budget(images, image_budget, key="caption")

    return PackedContext(
        graph_context=graph_str,
        graph_paths=list(graph_paths),
        text_chunks=kept_chunks,
        images=kept_images,
        conflicts=naive_conflicts(kept_chunks, graph_nodes),
    )


def naive_conflicts(chunks: Sequence[Any], graph_nodes: Sequence[str]) -> list[str]:
    """Original substring heuristic: flag 'not <node>' / 'unlike <node>' in chunk text.

    Kept for the old-vs-new comparison in Task 2.3; replaced by NLI there.
    """
    nodes = {n.lower() for n in graph_nodes}
    found: set[str] = set()
    for chunk in chunks:
        text = _get(chunk, "text").lower()
        for node in nodes:
            if f"not {node}" in text or f"unlike {node}" in text:
                found.add(f"Potential conflict: '{node}' mentioned negatively in text.")
    return sorted(found)
