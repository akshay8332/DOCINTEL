"""Pydantic + confidence validation (Stage 2 rules expand here)."""

from __future__ import annotations

from docintel.schemas.extraction import ExtractionResult, FieldPrediction


def apply_confidence_threshold(
    result: ExtractionResult,
    threshold: float = 0.75,
) -> ExtractionResult:
    fields: list[FieldPrediction] = []
    for field in result.fields:
        updated = field.model_copy(
            update={"needs_review": field.confidence < threshold}
        )
        fields.append(updated)
    return result.model_copy(update={"fields": fields})
