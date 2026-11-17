"""Phase 2a: structure-aware chunking, ported unchanged from the notebook.

Short pages stay whole. Dense pages (many lines or several bullets) are split
once near the middle at a blank line or bullet. Fragments under
``min_chunk_words`` are merged forward.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from graphrag.ingest.pdf_extract import PageRecord, write_json

BULLETS = ("•", "*", "-", "–", "→", "▪", "◦")  # noqa: RUF001 - en dash is a real bullet


@dataclass(frozen=True)
class ChunkingConfig:
    min_words: int = 40
    dense_line_count: int = 6
    min_chunk_words: int = 25


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_id: str
    page: int
    page_type: str
    source: str
    word_count: int


def is_bullet(line: str) -> bool:
    return line.strip().startswith(BULLETS)


def split_by_structure(text: str) -> list[str]:
    """Split once, searching upward from the middle for a blank line or bullet."""
    lines = text.splitlines()
    if len(lines) < 2:
        return [text]
    mid = len(lines) // 2
    split_at = mid
    for i in range(mid, 0, -1):
        if lines[i].strip() == "":
            split_at = i
            break
        if is_bullet(lines[i]) and i != mid:
            split_at = i
            break
    top = "\n".join(lines[:split_at]).strip()
    bottom = "\n".join(lines[split_at:]).strip()
    return [t for t in (top, bottom) if t]


def _merge_small(parts: list[str], min_chunk_words: int) -> list[str]:
    merged: list[str] = []
    buffer = ""
    for part in parts:
        if buffer:
            part = buffer + "\n" + part
            buffer = ""
        if len(part.split()) < min_chunk_words:
            buffer = part
        else:
            merged.append(part)
    if buffer:
        if merged:
            merged[-1] += "\n" + buffer
        else:
            merged.append(buffer)
    return merged


def make_chunk_id(doc_id: str, page: int, index: int) -> str:
    return f"{doc_id}_p{page:03d}_c{index:02d}"


def chunk_page(page: PageRecord, config: ChunkingConfig | None = None) -> list[Chunk]:
    cfg = config or ChunkingConfig()
    text = page.text.strip()
    word_count = len(text.split())

    if word_count < cfg.min_words:
        texts = [text]
    else:
        non_empty = [line for line in text.splitlines() if line.strip()]
        bullets = [line for line in non_empty if is_bullet(line)]
        is_dense = len(non_empty) > cfg.dense_line_count or len(bullets) >= 3
        parts = split_by_structure(text) if is_dense else [text]
        texts = _merge_small(parts, cfg.min_chunk_words)

    return [
        Chunk(
            chunk_id=make_chunk_id(page.pdf, page.page, i),
            text=chunk_text,
            doc_id=page.pdf,
            page=page.page,
            page_type=page.page_type,
            source="text",
            word_count=len(chunk_text.split()),
        )
        for i, chunk_text in enumerate(texts)
    ]


def chunk_pages(pages: list[PageRecord], config: ChunkingConfig | None = None) -> list[Chunk]:
    return [chunk for page in pages for chunk in chunk_page(page, config)]


def save_chunks(chunks: list[Chunk], documents_dir: Path) -> None:
    by_doc: dict[str, list[dict[str, object]]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(asdict(c))
    for doc_id, rows in by_doc.items():
        write_json(documents_dir / doc_id / "text_chunks.json", rows)


def load_chunks(documents_dir: Path) -> list[Chunk]:
    """Load chunks in the same deterministic order the original notebook used."""
    chunks: list[Chunk] = []
    for doc_dir in sorted(documents_dir.iterdir()):
        path = doc_dir / "text_chunks.json"
        if path.exists():
            chunks.extend(Chunk(**row) for row in json.loads(path.read_text(encoding="utf-8")))
    return chunks
