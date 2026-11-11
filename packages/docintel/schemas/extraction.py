"""Extraction result schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FieldPrediction(BaseModel):
    """Labeled span predicted by LayoutLM (or rule post-process)."""

    label: str
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    box: list[int] | None = None
    page_number: int = 1
    needs_review: bool = False
    start_token: int | None = None
    end_token: int | None = None


class ExtractionResult(BaseModel):
    document_id: str
    fields: list[FieldPrediction] = Field(default_factory=list)
    model_name: str | None = None
    checkpoint: str | None = None
    raw_token_labels: list[str] | None = None
