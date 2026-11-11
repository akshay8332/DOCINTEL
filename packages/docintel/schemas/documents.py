"""Document / page / word-box schemas (Pydantic v2)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class WordBox(BaseModel):
    """Single OCR/FUNSD word with axis-aligned box [x0, y0, x1, y1]."""

    text: str
    box: list[int] = Field(..., min_length=4, max_length=4)
    label: str | None = None

    @field_validator("box")
    @classmethod
    def _non_negative_box(cls, value: list[int]) -> list[int]:
        if any(v < 0 for v in value):
            raise ValueError("box coordinates must be non-negative")
        if value[2] < value[0] or value[3] < value[1]:
            raise ValueError("box must satisfy x1>=x0 and y1>=y0")
        return value


class Page(BaseModel):
    page_number: int = Field(ge=1, default=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    words: list[WordBox] = Field(default_factory=list)
    image_path: str | None = None


class DocumentMeta(BaseModel):
    document_id: str
    source: str = "funsd"
    split: Literal["train", "val", "test"] | None = None
    pages: list[Page] = Field(default_factory=list)
    annotation_path: str | None = None
