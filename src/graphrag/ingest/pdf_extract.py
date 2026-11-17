"""Phase 1: PDF text and image extraction with page classification."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import pymupdf

from graphrag.ingest.dedupe import ImageDeduplicator
from graphrag.log import get_logger

log = get_logger(__name__)

PageType = Literal["text", "slide", "code"]


def classify_page(text: str) -> PageType:
    """Heuristic page type: short-lined pages are slides, brace/def/import-heavy are code."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "text"
    avg_len = sum(len(line) for line in lines) / len(lines)
    if avg_len < 60 and len(lines) <= 12:
        return "slide"
    code_chars = text.count("{") + text.count("def ") + text.count("import ")
    if code_chars >= 3:
        return "code"
    return "text"


@dataclass
class PageRecord:
    pdf: str
    page: int
    text: str
    page_type: PageType


@dataclass
class ImageRecord:
    image_path: str
    doc_id: str
    page: int
    phash: str | None


@dataclass
class ExtractionStats:
    pdfs: int = 0
    pages: int = 0
    images_seen: int = 0
    duplicates: int = 0
    images_kept: int = 0
    page_types: Counter[str] = field(default_factory=Counter)


def extract_pdf(
    pdf_path: Path, image_dir: Path, deduper: ImageDeduplicator
) -> tuple[list[PageRecord], list[ImageRecord]]:
    """Extract page text and unique images from one PDF. ``doc_id`` is the filename."""
    doc_id = pdf_path.name
    image_dir.mkdir(parents=True, exist_ok=True)
    pages: list[PageRecord] = []
    images: list[ImageRecord] = []

    with pymupdf.open(pdf_path) as doc:  # type: ignore[no-untyped-call]
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            pages.append(PageRecord(doc_id, page_num, text, classify_page(text)))

            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    base = doc.extract_image(img[0])
                    data, ext = base["image"], base["ext"]
                except Exception:
                    continue
                result = deduper.check_and_register(data)
                if result.is_duplicate:
                    continue
                out = image_dir / f"p{page_num:03d}_img{img_index:02d}.{ext}"
                out.write_bytes(data)
                images.append(ImageRecord(out.as_posix(), doc_id, page_num, result.phash))
    return pages, images


def extract_corpus(
    corpus_dir: Path, documents_dir: Path, phash_threshold: int = 10
) -> ExtractionStats:
    """Extract every PDF in ``corpus_dir`` into ``documents_dir/<doc_id>/``."""
    pdfs = sorted(p for p in corpus_dir.iterdir() if p.suffix.lower() == ".pdf")
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {corpus_dir}")

    deduper = ImageDeduplicator(threshold=phash_threshold)
    stats = ExtractionStats(pdfs=len(pdfs))
    for pdf in pdfs:
        doc_out = documents_dir / pdf.name
        pages, images = extract_pdf(pdf, doc_out / "images", deduper)
        write_json(doc_out / "text_pages.json", [asdict(p) for p in pages])
        write_json(doc_out / "images_meta.json", [asdict(i) for i in images])
        stats.pages += len(pages)
        stats.images_kept += len(images)
        stats.page_types.update(p.page_type for p in pages)
        log.info("extracted_pdf", doc_id=pdf.name, pages=len(pages), images=len(images))

    stats.images_seen = deduper.checked
    stats.duplicates = deduper.duplicates
    return stats


def load_pages(documents_dir: Path) -> list[PageRecord]:
    return [PageRecord(**row) for row in _load_per_doc(documents_dir, "text_pages.json")]


def load_image_records(documents_dir: Path) -> list[ImageRecord]:
    return [ImageRecord(**row) for row in _load_per_doc(documents_dir, "images_meta.json")]


def _load_per_doc(documents_dir: Path, filename: str) -> list[dict[str, Any]]:
    if not documents_dir.exists():
        raise FileNotFoundError(f"{documents_dir} missing; run `graphrag ingest` first")
    rows: list[dict[str, Any]] = []
    for doc_dir in sorted(documents_dir.iterdir()):
        path = doc_dir / filename
        if path.exists():
            rows.extend(json.loads(path.read_text(encoding="utf-8")))
    return rows


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
