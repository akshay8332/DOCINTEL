"""Multi-hot targets. Every annotation that contains a word center stays on."""

from __future__ import annotations

import json
from pathlib import Path

from phase2.multilabel import annotation_retention, containing_fields, multilabel_matrix

from multilabel import FIELD_NAMES


def page_fields(doc, field_names: list[str] | None = None) -> list[dict]:
    names = set(field_names or FIELD_NAMES)
    page_index = int(doc.document_id.rsplit("-p", 1)[1])
    payload = json.loads(Path(doc.annotation_path).read_text(encoding="utf-8"))
    fields = []
    for field in payload.get("field_extractions") or []:
        if int(field.get("page", 0)) != page_index:
            continue
        name = str(field.get("fieldtype", ""))
        bbox = field.get("bbox")
        if name not in names or not bbox or len(bbox) != 4:
            continue
        box = [float(value) for value in bbox]
        fields.append(
            {
                "fieldtype": name,
                "text": field.get("text"),
                "bbox": [
                    min(box[0], box[2]),
                    min(box[1], box[3]),
                    max(box[0], box[2]),
                    max(box[1], box[3]),
                ],
            }
        )
    return fields


def relative_words(doc) -> list[tuple[str, list[float]]]:
    return [
        (word.text, [value / 1000.0 for value in word.box])
        for word in doc.pages[0].words
    ]


def word_targets(doc, field_names: list[str] | None = None) -> dict:
    """Nine binary values per OCR word, in ``field_names`` order."""
    names = list(field_names or FIELD_NAMES)
    fields = page_fields(doc, names)
    words = relative_words(doc)
    pairs = [(field["fieldtype"], field["bbox"]) for field in fields]
    matrix = multilabel_matrix(words, pairs, names)
    hits = containing_fields(words, pairs, allowed=set(names))
    covered = {index for word_hits in hits for index in word_hits}
    unmatched = [
        {
            "document_id": doc.document_id,
            "field": field["fieldtype"],
            "text": field["text"],
        }
        for index, field in enumerate(fields)
        if index not in covered
    ]
    positives = [0] * len(names)
    for row in matrix:
        for column, bit in enumerate(row):
            positives[column] += int(bit)
    return {
        "matrix": matrix,
        "fields": fields,
        "unmatched_fields": unmatched,
        "positive_words": positives,
        "words": len(matrix),
        "multi_label_words": sum(1 for row in matrix if sum(row) > 1),
        "retention": annotation_retention(words, pairs, allowed=set(names)),
    }


def pool_target_audit(pages: list[dict]) -> dict:
    names = list(FIELD_NAMES)
    positive = [0] * len(names)
    words = 0
    multi = 0
    unmatched = []
    retention = {}
    for page in pages:
        words += page["words"]
        multi += page["multi_label_words"]
        unmatched.extend(page["unmatched_fields"])
        for column, count in enumerate(page["positive_words"]):
            positive[column] += count
        for name, row in page["retention"].items():
            total = retention.setdefault(
                name, {"annotations": 0, "exclusive": 0, "multi": 0}
            )
            for key in total:
                total[key] += row[key]
    return {
        "words": words,
        "multi_label_words": multi,
        "positive_words": {name: positive[index] for index, name in enumerate(names)},
        "negative_words": {
            name: words - positive[index] for index, name in enumerate(names)
        },
        "unmatched_fields": len(unmatched),
        "unmatched_examples": unmatched[:20],
        "retention": retention,
    }
