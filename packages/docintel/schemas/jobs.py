"""Job and review queue schemas (Stage 2 stubs)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from docintel.schemas.extraction import FieldPrediction


class JobState(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"
    needs_review = "needs_review"


class JobCreate(BaseModel):
    source_uri: str | None = None
    filenames: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobStatus(BaseModel):
    job_id: str
    state: JobState = JobState.pending
    created_at: datetime | None = None
    updated_at: datetime | None = None
    fields: list[FieldPrediction] = Field(default_factory=list)
    error: str | None = None
    document_id: str | None = None


class ReviewItem(BaseModel):
    review_id: str
    job_id: str
    field: FieldPrediction
    resolved: bool = False
    corrected_value: str | None = None
