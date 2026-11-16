"""Compare an OCR word list with DocILE field annotations.

Both engines are passed in as words. This module does not select an engine.
"""

from __future__ import annotations

import re


def digits(text: str) -> str:
    return "".join(character for character in text if character.isdigit())


def normalize_text(text: str) -> str:
    cleaned = text.replace("&amp;", "&").replace("\n", " ")
    return re.sub(r"\s+", " ", cleaned).strip().casefold()


def _center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _contains(box: list[float], cx: float, cy: float) -> bool:
    return min(box[0], box[2]) <= cx <= max(box[0], box[2]) and min(box[1], box[3]) <= cy <= max(
        box[1], box[3]
    )


def _iou(left: list[float], right: list[float]) -> float:
    x0 = max(min(left[0], left[2]), min(right[0], right[2]))
    y0 = max(min(left[1], left[3]), min(right[1], right[3]))
    x1 = min(max(left[0], left[2]), max(right[0], right[2]))
    y1 = min(max(left[1], left[3]), max(right[1], right[3]))
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_left = abs(left[2] - left[0]) * abs(left[3] - left[1])
    area_right = abs(right[2] - right[0]) * abs(right[3] - right[1])
    union = area_left + area_right - inter
    return inter / union if union else 0.0


def words_inside(field_box: list[float], words: list[dict]) -> list[dict]:
    chosen = []
    for word in words:
        cx, cy = _center(word["box"])
        if _contains(field_box, cx, cy):
            chosen.append(word)
    return chosen


def _union(words: list[dict]) -> list[float]:
    boxes = [word["box"] for word in words]
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def compare_ocr(
    fields: list[dict],
    words: list[dict],
    *,
    amount_types: set[str],
    date_types: set[str],
) -> dict:
    """Score one page.

    ``fields`` items have ``fieldtype``, ``text`` and ``bbox``.
    ``words`` items have ``text`` and a relative ``box``.
    """
    detected = 0
    text_hits = 0
    amount_total = 0
    amount_hits = 0
    date_total = 0
    date_hits = 0
    ious: list[float] = []
    examples: list[dict] = []

    for field in fields:
        field_type = field["fieldtype"]
        inside = words_inside(field["bbox"], words)
        is_amount = field_type in amount_types
        is_date = field_type in date_types
        if is_amount:
            amount_total += 1
        if is_date:
            date_total += 1
        if not inside:
            if len(examples) < 6 and (is_amount or is_date):
                examples.append(
                    {
                        "field": field_type,
                        "annotation": field.get("text"),
                        "ocr": "",
                        "reason": "no_word_center",
                    }
                )
            continue
        detected += 1
        ocr_text = " ".join(word["text"] for word in inside)
        text_ok = normalize_text(ocr_text) == normalize_text(str(field.get("text") or ""))
        text_hits += int(text_ok)
        ious.append(_iou(_union(inside), field["bbox"]))
        gold_digits = digits(str(field.get("text") or ""))
        ocr_digits = digits(ocr_text)
        digits_match = bool(gold_digits) and ocr_digits == gold_digits
        if is_amount:
            amount_hits += int(digits_match)
        if is_date:
            date_hits += int(digits_match)
        if not text_ok and len(examples) < 6 and (is_amount or is_date):
            examples.append(
                {
                    "field": field_type,
                    "annotation": field.get("text"),
                    "ocr": ocr_text,
                    "reason": "text_mismatch",
                }
            )

    total = len(fields)

    def _rate(hits: int, count: int) -> float | None:
        return hits / count if count else None

    ordered = sorted(ious)
    median = None
    if ordered:
        mid = len(ordered) // 2
        median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {
        "fields": total,
        "detected_fields": detected,
        "text_exact": text_hits,
        "word_detection_recall": _rate(detected, total),
        "text_exact_on_covered": _rate(text_hits, detected),
        "text_exact_on_all_fields": _rate(text_hits, total),
        "median_box_iou": median,
        "amount_fields": amount_total,
        "amount_digit_hits": amount_hits,
        "amount_digit_recall": _rate(amount_hits, amount_total),
        "date_fields": date_total,
        "date_digit_hits": date_hits,
        "date_digit_recall": _rate(date_hits, date_total),
        "examples": examples,
    }


def merge_page_reports(pages: list[dict]) -> dict:
    """Pool page counts. Rates are recomputed from the sums."""
    count_keys = (
        "fields",
        "detected_fields",
        "text_exact",
        "amount_fields",
        "amount_digit_hits",
        "date_fields",
        "date_digit_hits",
    )
    sums = {key: sum(page[key] for page in pages) for key in count_keys}
    ious = [page["median_box_iou"] for page in pages if page["median_box_iou"] is not None]

    def _rate(hits: int, count: int) -> float | None:
        return hits / count if count else None

    return {
        "pages": len(pages),
        **sums,
        "word_detection_recall": _rate(sums["detected_fields"], sums["fields"]),
        "text_exact_on_covered": _rate(sums["text_exact"], sums["detected_fields"]),
        "text_exact_on_all_fields": _rate(sums["text_exact"], sums["fields"]),
        "mean_of_page_median_iou": sum(ious) / len(ious) if ious else None,
        "amount_digit_recall": _rate(sums["amount_digit_hits"], sums["amount_fields"]),
        "date_digit_recall": _rate(sums["date_digit_hits"], sums["date_fields"]),
    }
