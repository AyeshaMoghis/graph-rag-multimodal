"""Phase 3: image captions via Tesseract OCR, BLIP fallback, useless-caption filter.

Ported from the notebook. One behavioural addition: if the Tesseract binary is
missing, the notebook silently routed every image to BLIP. This module logs a
warning once so that the OCR/BLIP split in the stats can be trusted.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from graphrag.log import get_logger

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

log = get_logger(__name__)

_USELESS_PATTERNS = [
    r"^\[caption error",
    r"^[^a-zA-Z0-9]{0,5}$",
    r"^(a|the|an|image|photo|picture|figure|diagram|chart|graph|logo|icon|background|white|black|page)\s*\.?$",
    r"^[\W\d\s]+$",
]
USELESS_RE = re.compile("|".join(_USELESS_PATTERNS), re.IGNORECASE)


def is_useless(caption: str, min_words: int = 4) -> bool:
    """True if a caption carries no retrieval value."""
    c = caption.strip()
    return not c or len(c.split()) < min_words or bool(USELESS_RE.search(c))


def ocr_passes_threshold(text: str, min_alpha_words: int = 15) -> bool:
    """Count only words containing at least two letters, so symbol noise fails."""
    alpha_words = [w for w in text.split() if re.search(r"[a-zA-Z]{2,}", w)]
    return len(alpha_words) >= min_alpha_words


@dataclass
class CaptionRecord:
    image_id: str
    image_path: str
    caption: str
    method: Literal["ocr", "blip"]
    doc_id: str
    page: int


class Captioner:
    """OCR first, BLIP when OCR yields too little real text."""

    def __init__(
        self,
        blip_model_name: str = "Salesforce/blip-image-captioning-base",
        ocr_min_alpha_words: int = 15,
        ocr_fn: Callable[[PILImage], str] | None = None,
        blip_fn: Callable[[PILImage], str] | None = None,
    ) -> None:
        self.blip_model_name = blip_model_name
        self.ocr_min_alpha_words = ocr_min_alpha_words
        self._ocr_fn = ocr_fn or self._tesseract
        self._blip_fn = blip_fn or self._blip
        self._blip_cache: tuple[Any, Any, str] | None = None
        self._warned_no_tesseract = False

    def caption(self, image: PILImage) -> tuple[str, Literal["ocr", "blip"]]:
        ocr_text = self._ocr_fn(image)
        if ocr_passes_threshold(ocr_text, self.ocr_min_alpha_words):
            return ocr_text, "ocr"
        return self._blip_fn(image), "blip"

    def _tesseract(self, image: PILImage) -> str:
        import pytesseract

        try:
            return str(pytesseract.image_to_string(image, config="--psm 6")).strip()
        except pytesseract.TesseractNotFoundError:
            if not self._warned_no_tesseract:
                log.warning("tesseract_missing", effect="all images fall back to BLIP")
                self._warned_no_tesseract = True
            return ""
        except Exception:
            return ""

    def _blip(self, image: PILImage) -> str:
        try:
            import torch

            processor, model, device = self._load_blip()
            inputs = processor(image, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=80)
            return str(processor.decode(out[0], skip_special_tokens=True)).strip()
        except Exception as exc:
            return f"[caption error: {exc}]"

    def _load_blip(self) -> tuple[Any, Any, str]:
        if self._blip_cache is None:
            import torch
            from transformers import BlipForConditionalGeneration, BlipProcessor

            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = BlipProcessor.from_pretrained(self.blip_model_name)
            model = BlipForConditionalGeneration.from_pretrained(self.blip_model_name).to(device)  # type: ignore[arg-type]
            model.eval()
            self._blip_cache = (processor, model, device)
        return self._blip_cache


def image_id_for(doc_id: str, image_path: str) -> str:
    """Stable id from the document and image filename, e.g. ``lec1.pdf_p003_img00``."""
    return f"{doc_id}_{Path(image_path).stem}"
