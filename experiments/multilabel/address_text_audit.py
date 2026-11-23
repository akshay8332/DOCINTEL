"""Explain address exact-text misses. This does not train or change inference."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "experiments"))

from multilabel import FIELD_NAMES
from multilabel.ocr_text import correct_common_ocr

ADDRESS_FIELDS = ("vendor_address", "customer_billing_address")
CAUSES = (
    "missing_ocr",
    "ocr_character",
    "threshold",
    "line_grouping",
    "unwanted_nearby",
    "word_order",
    "other",
)
_TOKEN = re.compile(r"\S+")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text or "")


def norm_token(token: str) -> str:
    return re.sub(r"[^0-9A-Za-z]", "", token).casefold()


def edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if abs(len(left) - len(right)) > 2:
        return 3
    previous = list(range(len(right) + 1))
    for index, left_char in enumerate(left, start=1):
        current = [index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[right_index - 1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def box_iou(left: list[float], right: list[float]) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - inter
    return inter / union if union else 0.0


def center_inside(gold_box: list[float], word_box: list[int]) -> bool:
    cx = (word_box[0] + word_box[2]) / 2000
    cy = (word_box[1] + word_box[3]) / 2000
    return gold_box[0] <= cx <= gold_box[2] and gold_box[1] <= cy <= gold_box[3]


def _character_match(gold: str, ocr: str) -> bool:
    gold_norm = norm_token(gold)
    ocr_norm = norm_token(ocr)
    if not gold_norm or not ocr_norm or gold_norm == ocr_norm:
        return False
    if norm_token(correct_common_ocr(ocr)) == gold_norm:
        return True
    return edit_distance(gold_norm, ocr_norm) <= 2


def classify_address_field(
    gold: dict,
    words: list[dict],
    predictions: list[dict],
    sibling_golds: list[dict] | None = None,
) -> dict:
    """Assign one primary cause to one gold address, keeping every issue that applies."""
    siblings = sibling_golds or []
    in_box = [index for index, word in enumerate(words) if center_inside(gold["bbox"], word["box"])]
    used: set[int] = set()
    missing = []
    characters = []
    threshold = []
    separated = []
    matched: set[int] = set()
    left_out = []

    for token in tokens(gold["text"]):
        gold_norm = norm_token(token)
        if not gold_norm:
            continue
        exact = next(
            (
                index
                for index in in_box
                if index not in used and norm_token(words[index]["text"]) == gold_norm
            ),
            None,
        )
        if exact is not None:
            used.add(exact)
            matched.add(exact)
            continue
        character = next(
            (
                index
                for index in in_box
                if index not in used and _character_match(token, words[index]["text"])
            ),
            None,
        )
        if character is not None:
            used.add(character)
            matched.add(character)
            characters.append(
                {
                    "gold": token,
                    "ocr": words[character]["text"],
                    "distance": edit_distance(gold_norm, norm_token(words[character]["text"])),
                }
            )
            continue
        elsewhere = next(
            (
                index
                for index, word in enumerate(words)
                if index not in used and norm_token(word["text"]) == gold_norm
            ),
            None,
        )
        if elsewhere is None:
            missing.append({"gold": token, "on_page": False})
            continue
        used.add(elsewhere)
        matched.add(elsewhere)
        separated.append(elsewhere)

    same_type = [
        prediction
        for prediction in predictions
        if prediction["fieldtype"] == gold["fieldtype"] and prediction["page"] == gold["page"]
    ]
    ranked = sorted(
        ((box_iou(prediction["bbox"], gold["bbox"]), prediction) for prediction in same_type),
        key=lambda item: item[0],
        reverse=True,
    )
    overlap, prediction = ranked[0] if ranked else (0.0, None)
    located = prediction is not None and overlap >= 0.5
    predicted_indexes = set(prediction["word_indexes"]) if located else set()
    for index in list(matched):
        if index in predicted_indexes:
            continue
        word = words[index]
        evidence = {
            "text": word["text"],
            "probability": round(float(word["probability"]), 3),
            "truncated": bool(word["truncated"]),
            "inside_gold_box": index in in_box,
        }
        if word["probability"] < 0.5 or word["truncated"]:
            threshold.append(evidence)
        else:
            left_out.append(evidence)
    unwanted = []
    if located:
        for index in predicted_indexes:
            if index in matched:
                continue
            word = words[index]
            unwanted.append(
                {
                    "text": word["text"],
                    "probability": round(float(word["probability"]), 3),
                    "inside_gold_box": center_inside(gold["bbox"], word["box"]),
                }
            )
    split = located and sum(score >= 0.5 for score, _item in ranked) > 1
    merged = False
    if located:
        merged = sum(box_iou(prediction["bbox"], item["bbox"]) >= 0.5 for item in siblings) > 1
    predicted_text = prediction["text"] if located else None
    whitespace_only = False
    order_changed = False
    if predicted_text is not None and predicted_text != gold["text"]:
        gold_norms = [norm_token(token) for token in tokens(gold["text"])]
        pred_norms = [norm_token(token) for token in tokens(predicted_text)]
        gold_norms = [token for token in gold_norms if token]
        pred_norms = [token for token in pred_norms if token]
        whitespace_only = gold_norms == pred_norms
        order_changed = sorted(gold_norms) == sorted(pred_norms) and gold_norms != pred_norms
    explained = {
        "missing_ocr": len(missing),
        "ocr_character": len(characters),
        "threshold": len(threshold),
        "unwanted_nearby": len(unwanted),
        "line_grouping": int(split or merged or whitespace_only) + len(left_out),
        "word_order": int(order_changed and not whitespace_only),
    }
    exact = predicted_text == gold["text"]
    issues = [name for name, count in explained.items() if count]
    if exact:
        primary = "exact"
    else:
        primary = max(CAUSES[:-1], key=lambda name: (explained[name], -CAUSES.index(name)))
        if explained[primary] == 0:
            primary = "other"
    return {
        "document_id": gold["document_id"],
        "field": gold["fieldtype"],
        "page": gold["page"],
        "gold": gold["text"],
        "prediction": predicted_text,
        "overlap": round(overlap, 3),
        "score": None if prediction is None or not located else round(float(prediction["score"]), 3),
        "exact": exact,
        "primary": primary,
        "issues": issues,
        "missing_ocr": missing,
        "ocr_character": characters,
        "threshold": threshold,
        "unwanted_nearby": unwanted,
        "line_grouping": {
            "split": split,
            "merged": merged,
            "whitespace_only": whitespace_only,
            "left_out": left_out,
        },
        "word_order": order_changed and not whitespace_only,
        "unmatched_prediction": False,
    }


def unmatched_predictions(golds: list[dict], predictions: list[dict], cases: list[dict]) -> list[dict]:
    """Predictions that do not overlap any gold address of the same type."""
    consumed: set[int] = set()
    for case in cases:
        if case["overlap"] < 0.5:
            continue
        for index, prediction in enumerate(predictions):
            if (
                prediction["fieldtype"] == case["field"]
                and prediction["page"] == case["page"]
                and prediction["text"] == case["prediction"]
            ):
                consumed.add(index)
    extras = []
    for index, prediction in enumerate(predictions):
        if index in consumed or prediction["fieldtype"] not in ADDRESS_FIELDS:
            continue
        same = [
            gold
            for gold in golds
            if gold["fieldtype"] == prediction["fieldtype"] and gold["page"] == prediction["page"]
        ]
        if any(box_iou(prediction["bbox"], gold["bbox"]) >= 0.5 for gold in same):
            continue
        extras.append(
            {
                "document_id": prediction["document_id"],
                "field": prediction["fieldtype"],
                "page": prediction["page"],
                "gold": None,
                "prediction": prediction["text"],
                "overlap": 0.0,
                "score": round(float(prediction["score"]), 3),
                "exact": False,
                "primary": "unwanted_nearby",
                "issues": ["unwanted_nearby"],
                "missing_ocr": [],
                "ocr_character": [],
                "threshold": [],
                "unwanted_nearby": [{"text": prediction["text"], "inside_gold_box": False}],
                "line_grouping": {"split": False, "merged": False, "whitespace_only": False, "left_out": []},
                "word_order": False,
                "unmatched_prediction": True,
            }
        )
    return extras


def build_report(cases: list[dict], *, checkpoint: str, pages: int, documents: int) -> dict:
    fields = {}
    examples = {}
    for name in ADDRESS_FIELDS:
        subset = [case for case in cases if case["field"] == name and not case["unmatched_prediction"]]
        extras = [case for case in cases if case["field"] == name and case["unmatched_prediction"]]
        mismatches = [case for case in subset if not case["exact"]] + extras
        issues: Counter[str] = Counter()
        truncated = 0
        for case in mismatches:
            issues.update(case["issues"])
            if any(item.get("truncated") for item in case["threshold"]):
                truncated += 1
        fields[name] = {
            "gold": len(subset),
            "exact_text": sum(case["exact"] for case in subset),
            "text_mismatch_at_location": sum(
                case["overlap"] >= 0.5 and not case["exact"] for case in subset
            ),
            "missed_location": sum(case["overlap"] < 0.5 for case in subset),
            "unmatched_predictions": len(extras),
            "primary_cause": dict(Counter(case["primary"] for case in mismatches)),
            "issue_present": dict(issues),
            "threshold_includes_truncated_word": truncated,
        }
        examples[name] = {}
        for cause in CAUSES:
            if cause == "other":
                continue
            chosen = [case for case in mismatches if case["primary"] == cause]
            chosen.sort(key=lambda case: (case["overlap"] < 0.5, -case["overlap"]))
            examples[name][cause] = [_example(case) for case in chosen[:3]]
    return {
        "checkpoint": checkpoint,
        "checkpoint_updated": False,
        "validation_pages": pages,
        "documents": documents,
        "threshold": 0.5,
        "grouping": "line-block-v2",
        "annotations_changed": False,
        "ocr_changed": False,
        "fields": fields,
        "examples": examples,
        "cases": cases,
    }


def _example(case: dict) -> dict:
    return {
        "document_id": case["document_id"],
        "page": case["page"],
        "overlap": case["overlap"],
        "score": case["score"],
        "gold": case["gold"],
        "prediction": case["prediction"],
        "issues": case["issues"],
        "missing_ocr": case["missing_ocr"],
        "ocr_character": case["ocr_character"],
        "threshold": case["threshold"],
        "unwanted_nearby": case["unwanted_nearby"],
        "line_grouping": case["line_grouping"],
        "word_order": case["word_order"],
    }


def render_summary(report: dict) -> str:
    lines = [
        "# Address text audit",
        "",
        "Validation pages only. The checkpoint, OCR, threshold, grouping, and annotations were not changed.",
        "",
        f"Checkpoint: `{report['checkpoint']}`",
        f"Pages: {report['validation_pages']}. Documents: {report['documents']}. Threshold 0.5. Grouping line-block-v2.",
        "",
        "Primary cause counts each gold address that is not an exact text match, plus predictions that overlap no gold address.",
        "A field can also carry other issues; those are counted under issue present.",
        "",
        "| Field | Gold | Exact text | Right place, wrong text | Missed place | Extra predictions |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ADDRESS_FIELDS:
        row = report["fields"][name]
        lines.append(
            f"| {name} | {row['gold']} | {row['exact_text']} | {row['text_mismatch_at_location']} | "
            f"{row['missed_location']} | {row['unmatched_predictions']} |"
        )
    lines.extend(["", "## Primary cause", "", "| Field | " + " | ".join(CAUSES) + " |", "| --- | " + " | ".join("---:" for _ in CAUSES) + " |"])
    for name in ADDRESS_FIELDS:
        causes = report["fields"][name]["primary_cause"]
        lines.append("| " + name + " | " + " | ".join(str(causes.get(cause, 0)) for cause in CAUSES) + " |")
    lines.extend(["", "## Other issues present on the same mismatch", ""])
    for name in ADDRESS_FIELDS:
        present = report["fields"][name]["issue_present"]
        rendered = ", ".join(f"{cause} {present[cause]}" for cause in CAUSES if present.get(cause))
        truncated = report["fields"][name]["threshold_includes_truncated_word"]
        lines.append(f"- {name}: {rendered or 'none'}. Threshold rows that include a word past the 512-token window: {truncated}.")
    lines.extend(["", "## Examples", ""])
    for name in ADDRESS_FIELDS:
        lines.append(f"### {name}")
        lines.append("")
        for cause in CAUSES:
            if cause == "other":
                continue
            samples = report["examples"][name].get(cause) or []
            if not samples:
                continue
            lines.append(f"#### {cause}")
            lines.append("")
            for sample in samples:
                lines.append(f"- `{sample['document_id']}` page {sample['page']} overlap {sample['overlap']}")
                lines.append(f"  - gold: {sample['gold']!r}")
                lines.append(f"  - prediction: {sample['prediction']!r}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _gold_fields(doc) -> list[dict]:
    page = int(doc.document_id.rsplit("-p", 1)[1])
    payload = json.loads(Path(doc.annotation_path).read_text(encoding="utf-8"))
    fields = []
    for field in payload.get("field_extractions") or []:
        if field.get("fieldtype") not in ADDRESS_FIELDS or int(field.get("page", 0)) != page:
            continue
        box = [float(value) for value in field["bbox"]]
        fields.append(
            {
                "document_id": doc.document_id.rsplit("-p", 1)[0],
                "page": page,
                "fieldtype": field["fieldtype"],
                "text": field.get("text") or "",
                "bbox": [min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])],
            }
        )
    return fields


def _page_words(page: dict, probabilities: list[list[float]], column: int) -> list[dict]:
    seen = {word_id for word_id in page["word_ids"] if word_id is not None}
    rows = []
    for index, text in enumerate(page["words"]):
        probability = 0.0
        if index < len(probabilities) and column < len(probabilities[index]):
            probability = float(probabilities[index][column])
        rows.append(
            {
                "text": text,
                "box": [int(value) for value in page["boxes"][index]],
                "probability": probability,
                "truncated": index not in seen,
            }
        )
    return rows


def audit_documents(docs, model, processor, device: str, max_length: int) -> list[dict]:
    import torch

    from multilabel.align import word_probabilities
    from multilabel.data import encode_document
    from multilabel.group import group_fields

    cases = []
    for count, doc in enumerate(docs, start=1):
        page = encode_document(doc, processor, max_length=max_length)
        batch = {
            key: page[key].unsqueeze(0).to(device)
            for key in ("input_ids", "attention_mask", "bbox", "pixel_values")
        }
        with torch.inference_mode():
            logits, _loss = model(**batch)
        probabilities = word_probabilities(
            torch.sigmoid(logits[0]).cpu().tolist(),
            page["word_ids"],
            len(page["words"]),
        )
        predicted = group_fields(
            page["words"],
            page["boxes"],
            probabilities,
            FIELD_NAMES,
            threshold=0.5,
            page_index=page["page_index"],
            grouping="line-block-v2",
        )
        predictions = [
            {
                "document_id": doc.document_id.rsplit("-p", 1)[0],
                "fieldtype": field.label,
                "text": field.text,
                "bbox": field.box_relative,
                "score": field.confidence,
                "word_indexes": field.word_indexes,
                "page": field.page_index,
            }
            for field in predicted
            if field.label in ADDRESS_FIELDS
        ]
        golds = _gold_fields(doc)
        page_cases = []
        for gold in golds:
            column = FIELD_NAMES.index(gold["fieldtype"])
            siblings = [
                item
                for item in golds
                if item is not gold and item["fieldtype"] == gold["fieldtype"] and item["page"] == gold["page"]
            ]
            page_cases.append(
                classify_address_field(
                    gold,
                    _page_words(page, probabilities, column),
                    predictions,
                    siblings,
                )
            )
        cases.extend(page_cases)
        cases.extend(unmatched_predictions(golds, predictions, page_cases))
        del page, batch, logits, probabilities
        if count % 20 == 0:
            print(f"audited {count}", flush=True)
    return cases


def main() -> int:
    from docintel.data.docile import load_docile_pages
    from multilabel.checkpoint import assert_diagnostic_destination, assert_output_allowed
    from multilabel.evaluate import load_multilabel

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Existing page images. The default renders beside the output file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "diagnostics" / "address-text-audit-100.json",
    )
    args = parser.parse_args()
    summary = args.output.with_suffix(".md")
    assert_diagnostic_destination(args.output)
    assert_diagnostic_destination(summary)
    assert_output_allowed(args.output)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Reading checkpoint {args.checkpoint}", flush=True)
    model, processor, _payload = load_multilabel(args.checkpoint, device)
    image_dir = args.image_dir or (args.output.parent / "page_images")
    docs = load_docile_pages(args.data_dir, "val", limit=args.limit, image_dir=image_dir)
    print(f"Loaded pages {len(docs)}", flush=True)
    cases = audit_documents(docs, model, processor, device, args.max_length)
    documents = len({doc.document_id.rsplit("-p", 1)[0] for doc in docs})
    report = build_report(
        cases,
        checkpoint=str(args.checkpoint),
        pages=len(docs),
        documents=documents,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary.write_text(render_summary(report), encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)
    print(f"Wrote {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
