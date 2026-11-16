"""Group positive words into field instances and export DocILE boxes."""

from __future__ import annotations

from dataclasses import dataclass


def to_relative_box(box: list[int] | list[float]) -> list[float]:
    relative = [max(0.0, min(1.0, float(value) / 1000.0)) for value in box]
    if relative[2] < relative[0]:
        relative[0], relative[2] = relative[2], relative[0]
    if relative[3] < relative[1]:
        relative[1], relative[3] = relative[3], relative[1]
    return relative


def _union(boxes: list[list[int]]) -> list[int]:
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _vertically_far(span_box: list[int], word_box: list[int], gap: int) -> bool:
    return word_box[1] > span_box[3] + gap


MULTILINE_FIELDS = {
    "vendor_name",
    "customer_billing_name",
    "vendor_address",
    "customer_billing_address",
}

# Weights and post-processing are versioned separately.
# reading-order-v1 produced the first sigmoid scores. line-block-v2 is the
# inference baseline used with the frozen epoch-8 checkpoint.
GROUPING_READING_ORDER = "reading-order-v1"
GROUPING_LINE_BLOCK = "line-block-v2"
INFERENCE_BASELINE = GROUPING_LINE_BLOCK


def _same_line(line_box: list[int], word_box: list[int]) -> bool:
    overlap = min(line_box[3], word_box[3]) - max(line_box[1], word_box[1])
    height = max(1, min(line_box[3] - line_box[1], word_box[3] - word_box[1]))
    return overlap >= 0.4 * height


def _separate_columns(previous: list[int], index: int, boxes: list[list[int]]) -> bool:
    """Split two wide labels that sit side by side, not ordinary word spaces."""
    gap = boxes[index][0] - boxes[previous][2]
    previous_width = boxes[previous][2] - boxes[previous][0]
    width = boxes[index][2] - boxes[index][0]
    return gap > 25 and previous_width >= 80 and width >= 80


def _line_segments(indexes: list[int], boxes: list[list[int]]) -> list[list[int]]:
    ordered = sorted(indexes, key=lambda i: ((boxes[i][1] + boxes[i][3]) / 2, boxes[i][0]))
    lines: list[list[int]] = []
    for index in ordered:
        if not lines or not _same_line(_union([boxes[i] for i in lines[-1]]), boxes[index]):
            lines.append([index])
        else:
            lines[-1].append(index)
    segments: list[list[int]] = []
    for line in lines:
        line.sort(key=lambda i: boxes[i][0])
        current = [line[0]]
        for index in line[1:]:
            if _separate_columns(current[-1], index, boxes):
                segments.append(current)
                current = [index]
            else:
                current.append(index)
        segments.append(current)
    return segments


def _multiline_spans(indexes: list[int], boxes: list[list[int]], block_gap: int) -> list[list[int]]:
    spans: list[list[int]] = []
    current: list[int] = []
    for segment in _line_segments(indexes, boxes):
        if not current:
            current = list(segment)
            continue
        span_box = _union([boxes[i] for i in current])
        segment_box = _union([boxes[i] for i in segment])
        overlaps_x = min(span_box[2], segment_box[2]) - max(span_box[0], segment_box[0]) > 0
        if segment_box[1] - span_box[3] <= block_gap and overlaps_x:
            current.extend(segment)
        else:
            spans.append(current)
            current = list(segment)
    if current:
        spans.append(current)
    return spans


def _reading_order_spans(
    indexes: list[int],
    boxes: list[list[int]],
    spatial_gap: int,
) -> list[list[int]]:
    """Original grouping. A non-positive word between two positives starts a new span."""
    spans: list[list[int]] = []
    current: list[int] = []
    for index in indexes:
        if not current:
            current = [index]
            continue
        previous = current[-1]
        span_box = _union([boxes[item] for item in current])
        split = index != previous + 1 or _vertically_far(span_box, boxes[index], spatial_gap)
        if split:
            spans.append(current)
            current = [index]
        else:
            current.append(index)
    if current:
        spans.append(current)
    return spans


def assemble_text(words: list[str], boxes: list[list[int]]) -> str:
    """Join words on one visual line with spaces and lines with newlines."""
    if not words:
        return ""
    order = sorted(range(len(words)), key=lambda i: ((boxes[i][1] + boxes[i][3]) / 2, boxes[i][0]))
    lines: list[list[int]] = []
    for index in order:
        top, bottom = boxes[index][1], boxes[index][3]
        if not lines:
            lines.append([index])
            continue
        line = lines[-1]
        line_top = min(boxes[i][1] for i in line)
        line_bottom = max(boxes[i][3] for i in line)
        overlap = min(line_bottom, bottom) - max(line_top, top)
        height = max(1, min(line_bottom - line_top, bottom - top))
        if overlap >= 0.4 * height:
            line.append(index)
        else:
            lines.append([index])
    return "\n".join(
        " ".join(words[index] for index in sorted(line, key=lambda i: boxes[i][0]))
        for line in lines
    )


@dataclass(frozen=True)
class PredictedField:
    label: str
    text: str
    confidence: float
    box_0_1000: list[int]
    box_relative: list[float]
    page_index: int
    word_indexes: tuple[int, ...]

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


def group_fields(
    words: list[str],
    boxes: list[list[int]],
    probabilities: list[list[float]],
    field_names: list[str],
    *,
    threshold: float = 0.5,
    page_index: int = 0,
    spatial_gap: int = 30,
    block_gap: int = 40,
    grouping: str = INFERENCE_BASELINE,
    field_thresholds: list[float] | None = None,
) -> list[PredictedField]:
    """One span per run of nearby positive words. A gap starts another instance.

    ``line-block-v2`` keeps a name or address together across line breaks.
    ``reading-order-v1`` splits when the next positive word is not the next OCR word.
    Amounts, dates, and ids use reading order in both versions.
    """
    if grouping not in (GROUPING_READING_ORDER, GROUPING_LINE_BLOCK):
        raise ValueError(f"Unknown grouping {grouping}")
    predicted: list[PredictedField] = []
    for column, name in enumerate(field_names):
        cutoff = field_thresholds[column] if field_thresholds is not None else threshold
        active = [
            index
            for index, row in enumerate(probabilities)
            if column < len(row) and row[column] >= cutoff
        ]
        use_lines = grouping == GROUPING_LINE_BLOCK and name in MULTILINE_FIELDS
        if use_lines:
            spans = _multiline_spans(active, boxes, block_gap)
        else:
            spans = _reading_order_spans(active, boxes, spatial_gap)
        for span in spans:
            chosen = [boxes[index] for index in span]
            scores = [probabilities[index][column] for index in span]
            predicted.append(
                PredictedField(
                    label=name,
                    text=(
                        assemble_text([words[index] for index in span], [boxes[index] for index in span])
                        if grouping == GROUPING_LINE_BLOCK
                        else " ".join(words[index] for index in span)
                    ),
                    confidence=sum(scores) / len(scores),
                    box_0_1000=_union(chosen),
                    box_relative=to_relative_box(_union(chosen)),
                    page_index=page_index,
                    word_indexes=tuple(span),
                )
            )
    return predicted
