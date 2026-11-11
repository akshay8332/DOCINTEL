"""OCR / PDF rasterize stub (Stage 2). FUNSD Stage 1 uses provided words."""

from __future__ import annotations

from pathlib import Path

from docintel.schemas.documents import DocumentMeta, Page


def pdf_to_images(pdf_path: Path, *, dpi: int = 200) -> list[Path]:
    raise NotImplementedError("PDF rasterize is Stage 2 — install pdf2image/poppler later.")


def run_ocr(image_path: Path) -> DocumentMeta:
    """Stub OCR returning an empty page shell."""
    return DocumentMeta(
        document_id=image_path.stem,
        source="ocr-stub",
        pages=[Page(page_number=1, width=1000, height=1000, words=[], image_path=str(image_path))],
    )
