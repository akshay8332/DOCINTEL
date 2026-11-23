"""Compare address predictions with the DocILE OCR words inside the gold box.

Uses the saved audit and the local DocILE files. It does not load the checkpoint
or replace the OCR.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("vendor_address", "customer_billing_address")


def _norm_seq(text: str) -> list[str]:
    from multilabel.address_text_audit import norm_token, tokens

    return [token for token in (norm_token(part) for part in tokens(text or "")) if token]


def _same(left: str, right: str, normalized: bool) -> bool:
    if normalized:
        return _norm_seq(left) == _norm_seq(right)
    return (left or "") == (right or "")


def _gold_box(data_dir: Path, case: dict) -> list[float] | None:
    payload = json.loads((data_dir / "annotations" / f"{case['document_id']}.json").read_text(encoding="utf-8"))
    for field in payload.get("field_extractions") or []:
        if field.get("fieldtype") != case["field"] or int(field.get("page", 0)) != case["page"]:
            continue
        if (field.get("text") or "") != (case.get("gold") or ""):
            continue
        box = [float(value) for value in field["bbox"]]
        return [min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])]
    return None


def _ocr_text(data_dir: Path, case: dict, gold_box: list[float]) -> str:
    from docintel.data.docile import _page_words, layout_box
    from multilabel.address_text_audit import center_inside
    from multilabel.group import assemble_text

    payload = json.loads((data_dir / "ocr" / f"{case['document_id']}.json").read_text(encoding="utf-8"))
    page = next(item for item in payload.get("pages") or [] if int(item.get("page_idx", 0)) == case["page"])
    words: list[str] = []
    boxes: list[list[int]] = []
    for text, box in _page_words(page):
        laid = layout_box(box)
        if center_inside(gold_box, laid):
            words.append(text)
            boxes.append(laid)
    return assemble_text(words, boxes)


def classify(case: dict, ocr_text: str) -> dict:
    prediction = case.get("prediction") or ""
    gold = case.get("gold") or ""
    return {
        "document_id": case["document_id"],
        "page": case["page"],
        "overlap": case["overlap"],
        "gold": gold,
        "prediction": case.get("prediction"),
        "ocr_inside_box": ocr_text,
        "strict": {
            "ocr_equals_gold": _same(ocr_text, gold, False),
            "prediction_equals_gold": _same(prediction, gold, False),
            "prediction_equals_ocr": _same(prediction, ocr_text, False),
        },
        "normalized": {
            "ocr_equals_gold": _same(ocr_text, gold, True),
            "prediction_equals_gold": _same(prediction, gold, True),
            "prediction_equals_ocr": _same(prediction, ocr_text, True),
        },
    }


def _bucket(row: dict, normalized: bool) -> str:
    key = "normalized" if normalized else "strict"
    flags = row[key]
    if flags["prediction_equals_gold"]:
        return "prediction_matches_gold"
    if flags["ocr_equals_gold"] and flags["prediction_equals_ocr"]:
        return "ocr_matches_gold_prediction_matches_ocr"
    if flags["prediction_equals_ocr"]:
        return "prediction_follows_ocr"
    if flags["ocr_equals_gold"]:
        return "ocr_matches_gold_prediction_differs"
    return "neither_matches"


def build_report(audit: dict, data_dir: Path) -> dict:
    fields = {}
    examples = {}
    for name in FIELDS:
        cases = [
            case
            for case in audit["cases"]
            if case["field"] == name and not case["unmatched_prediction"] and case["overlap"] >= 0.5
        ]
        rows = []
        missing_box = 0
        for case in cases:
            box = _gold_box(data_dir, case)
            if box is None:
                missing_box += 1
                continue
            rows.append(classify(case, _ocr_text(data_dir, case, box)))
        wrong = [row for row in rows if not row["strict"]["prediction_equals_gold"]]
        fields[name] = {
            "located": len(rows),
            "missing_gold_box": missing_box,
            "strict": dict(Counter(_bucket(row, False) for row in rows)),
            "normalized": dict(Counter(_bucket(row, True) for row in wrong)),
            "located_wrong_text": len(wrong),
        }
        examples[name] = {}
        for label in ("prediction_follows_ocr", "ocr_matches_gold_prediction_differs"):
            chosen = [row for row in wrong if _bucket(row, True) == label]
            chosen.sort(key=lambda row: -row["overlap"])
            examples[name][label] = [
                {
                    "document_id": row["document_id"],
                    "page": row["page"],
                    "overlap": row["overlap"],
                    "gold": row["gold"],
                    "ocr_inside_box": row["ocr_inside_box"],
                    "prediction": row["prediction"],
                }
                for row in chosen[:3]
            ]
    return {
        "source": "address-text-audit-100.json",
        "validation_pages": audit["validation_pages"],
        "documents": audit["documents"],
        "ocr_replaced": False,
        "checkpoint_updated": False,
        "fields": fields,
        "examples": examples,
    }


def render_report(report: dict) -> str:
    lines = [
        "# Address OCR wording",
        "",
        "Located addresses on the 100 validation pages. The text inside each gold box is the DocILE OCR, assembled with the same line breaks as inference.",
        "DocILE OCR was not replaced, and the checkpoint was not changed.",
        "",
        "Normalized comparison ignores punctuation and line breaks. Strict comparison is the official exact string.",
        "",
        "## Located addresses, strict string",
        "",
        "| Field | Located | Prediction matches gold | Prediction follows OCR, gold differs | OCR matches gold, prediction differs | Neither |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    order = (
        "prediction_matches_gold",
        "prediction_follows_ocr",
        "ocr_matches_gold_prediction_differs",
        "neither_matches",
    )
    for name in FIELDS:
        counts = report["fields"][name]["strict"]
        lines.append(
            "| {name} | {total} | {gold} | {follows} | {differs} | {neither} |".format(
                name=name,
                total=report["fields"][name]["located"],
                gold=counts.get(order[0], 0),
                follows=counts.get(order[1], 0),
                differs=counts.get(order[2], 0),
                neither=counts.get(order[3], 0),
            )
        )
    lines.extend(
        [
            "",
            "## Located wrong text, normalized tokens",
            "",
            "| Field | Wrong text | Matches gold after punctuation | Prediction follows OCR | OCR matches gold, prediction differs | Neither |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in FIELDS:
        counts = report["fields"][name]["normalized"]
        lines.append(
            "| {name} | {total} | {gold} | {follows} | {differs} | {neither} |".format(
                name=name,
                total=report["fields"][name]["located_wrong_text"],
                gold=counts.get("prediction_matches_gold", 0),
                follows=counts.get("prediction_follows_ocr", 0),
                differs=counts.get("ocr_matches_gold_prediction_differs", 0),
                neither=counts.get("neither_matches", 0),
            )
        )
    lines.extend(["", "## Examples", ""])
    for name in FIELDS:
        lines.append(f"### {name}")
        lines.append("")
        for label in ("prediction_follows_ocr", "ocr_matches_gold_prediction_differs"):
            samples = report["examples"][name][label]
            if not samples:
                continue
            lines.append(f"#### {label}")
            lines.append("")
            for sample in samples:
                lines.append(f"- `{sample['document_id']}` page {sample['page']} overlap {sample['overlap']}")
                lines.append(f"  - gold: {sample['gold']!r}")
                lines.append(f"  - ocr: {sample['ocr_inside_box']!r}")
                lines.append(f"  - prediction: {sample['prediction']!r}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "docile")
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "artifacts" / "diagnostics" / "address-text-audit-100.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "diagnostics" / "address-ocr-wording-100.json",
    )
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    report = build_report(audit, args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = args.output.with_suffix(".md")
    summary.write_text(render_report(report), encoding="utf-8")
    print(summary.read_text(encoding="utf-8"))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
