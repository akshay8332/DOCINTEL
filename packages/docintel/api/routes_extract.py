"""Sync extract + review stubs (Stage 2)."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from docintel.schemas.extraction import ExtractionResult
from docintel.schemas.jobs import ReviewItem

router = APIRouter(prefix="/v1", tags=["extract"])


class SyncExtractRequest(BaseModel):
    document_id: str
    # Stage 2: multipart upload or storage URI
    note: str = "stub — wire services.extract"


@router.post("/extract/sync", response_model=ExtractionResult)
def extract_sync(body: SyncExtractRequest) -> ExtractionResult:
    return ExtractionResult(document_id=body.document_id, fields=[])


@router.get("/review", response_model=list[ReviewItem])
def list_review() -> list[ReviewItem]:
    return []


@router.post("/review/{review_id}/resolve")
def resolve_review(review_id: str, corrected_value: str | None = None) -> dict:
    return {"review_id": review_id, "resolved": True, "corrected_value": corrected_value}
