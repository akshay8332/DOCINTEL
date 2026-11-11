"""Extraction service — wraps LayoutLM inference (shared by API + workers)."""

from __future__ import annotations

from pathlib import Path

from docintel.data.transforms import document_to_example
from docintel.models.layoutlm import LayoutLMBundle, infer_spans
from docintel.schemas.documents import DocumentMeta
from docintel.schemas.extraction import ExtractionResult


def extract_document(
    doc: DocumentMeta,
    bundle: LayoutLMBundle,
    *,
    confidence_threshold: float = 0.75,
    checkpoint: str | Path | None = None,
) -> ExtractionResult:
    example = document_to_example(doc, label_list=bundle.label_list)
    result = infer_spans(
        bundle, example, confidence_threshold=confidence_threshold
    )
    if checkpoint:
        result.checkpoint = str(checkpoint)
    return result
