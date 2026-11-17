from __future__ import annotations

from graphrag.ingest.chunking import (
    ChunkingConfig,
    chunk_page,
    is_bullet,
    make_chunk_id,
    split_by_structure,
)
from graphrag.ingest.pdf_extract import PageRecord, classify_page


def page(text: str, n: int = 3) -> PageRecord:
    return PageRecord(pdf="lec.pdf", page=n, text=text, page_type="text")


def words(n: int, word: str = "word") -> str:
    return " ".join([word] * n)


def test_chunk_id_format() -> None:
    assert make_chunk_id("lec.pdf", 3, 1) == "lec.pdf_p003_c01"


def test_short_page_stays_whole() -> None:
    chunks = chunk_page(page(words(10)))
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "lec.pdf_p003_c00"
    assert chunks[0].word_count == 10


def test_empty_page_still_yields_one_chunk_like_the_original() -> None:
    chunks = chunk_page(page(""))
    assert len(chunks) == 1 and chunks[0].text == ""


def test_dense_page_splits_at_blank_line_near_middle() -> None:
    top = "\n".join(words(7, "alpha") for _ in range(4))
    bottom = "\n".join(words(7, "beta") for _ in range(4))
    chunks = chunk_page(page(top + "\n\n" + bottom))
    assert len(chunks) == 2
    assert "beta" not in chunks[0].text
    assert "alpha" not in chunks[1].text


def test_small_fragment_is_merged_forward() -> None:
    # Dense (> 6 lines), but the bottom half is under 25 words, so it merges back.
    text = "\n".join(words(8, "alpha") for _ in range(5)) + "\n\n" + "\n".join(["tiny"] * 3)
    chunks = chunk_page(page(text))
    assert len(chunks) == 1
    assert "tiny" in chunks[0].text


def test_non_dense_long_page_is_not_split() -> None:
    text = "\n".join(words(20) for _ in range(3))
    assert len(chunk_page(page(text), ChunkingConfig())) == 1


def test_split_by_structure_single_line() -> None:
    assert split_by_structure("one line") == ["one line"]


def test_is_bullet() -> None:
    assert is_bullet("  • point") and is_bullet("- dash") and not is_bullet("plain")


def test_classify_page() -> None:
    assert classify_page("") == "text"
    assert classify_page("Title\nShort bullet\nAnother") == "slide"
    code = "x = 1 " * 40 + "\nimport os\nimport re\ndef f(): {\n" + "y " * 200
    assert classify_page(code) == "code"
    prose = "\n".join(
        "This is a long line of ordinary prose text that goes on and on and on." for _ in range(15)
    )
    assert classify_page(prose) == "text"
