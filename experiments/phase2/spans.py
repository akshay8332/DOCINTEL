"""Build field spans and DocILE relative boxes from word tags."""

from __future__ import annotations

from dataclasses import dataclass

from phase2.bio_decode import DecodedSpan, decode_bio_labels


@dataclass(frozen=True)
class FieldSpan:
    label: str
    text: str
    confidence: float
    box_0_1000: list[int]
    box_relative: list[float]
    page_index: int
    start: int
    end: int
    recovered: bool

    def to_docile_dict(self) -> dict:
        return {
            "bbox": self.box_relative,
            "page": self.page_index,
            "score": self.confidence,
            "text": self.text,
            "fieldtype": self.label,
            "line_item_id": None,
            "use_only_for_ap": False,
        }


def to_relative_box(box: list[int] | list[float]) -> list[float]:
    """Convert LayoutLM's 0–1000 box to DocILE's 0–1 box."""
    relative = [max(0.0, min(1.0, float(value) / 1000.0)) for value in box]
    if relative[2] < relative[0]:
        relative[0], relative[2] = relative[2], relative[0]
    if relative[3] < relative[1]:
        relative[1], relative[3] = relative[3], relative[1]
    return relative


def union_box(boxes: list[list[int]]) -> list[int]:
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def spans_from_word_labels(
    words: list[str],
    labels: list[str],
    boxes: list[list[int]],
    confidences: list[float] | None = None,
    *,
    page_index: int = 0,
) -> list[FieldSpan]:
    """Group words with the robust BIO rules and union their boxes."""
    if confidences is None:
        confidences = [1.0] * len(words)
    fields: list[FieldSpan] = []
    for span in decode_bio_labels(labels):
        fields.append(_span_to_field(words, boxes, confidences, span, page_index))
    return fields


def _span_to_field(
    words: list[str],
    boxes: list[list[int]],
    confidences: list[float],
    span: DecodedSpan,
    page_index: int,
) -> FieldSpan:
    chosen = boxes[span.start : span.end]
    box = union_box(chosen)
    scores = confidences[span.start : span.end]
    confidence = sum(scores) / len(scores)
    return FieldSpan(
        label=span.label,
        text=" ".join(words[span.start : span.end]),
        confidence=float(confidence),
        box_0_1000=box,
        box_relative=to_relative_box(box),
        page_index=page_index,
        start=span.start,
        end=span.end - 1,
        recovered=span.recovered,
    )
