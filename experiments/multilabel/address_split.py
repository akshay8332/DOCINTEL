"""Split located address misses into OCR wording, an extra header, or both.

Reads the saved audit. It does not load the checkpoint or change inference.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("vendor_address", "customer_billing_address")


def split_label(case: dict) -> str:
    """Label one gold address that overlaps the prediction but is not exact."""
    outside = [word for word in case["unwanted_nearby"] if not word.get("inside_gold_box", False)]
    inside = [word for word in case["unwanted_nearby"] if word.get("inside_gold_box", False)]
    ocr_wording = bool(case["missing_ocr"] or case["ocr_character"] or inside)
    extra_header = bool(outside)
    if ocr_wording and extra_header:
        return "both"
    if ocr_wording:
        return "ocr_wording"
    if extra_header:
        return "extra_header"
    grouping = case.get("line_grouping") or {}
    if grouping.get("whitespace_only") or grouping.get("split") or grouping.get("merged"):
        return "line_break"
    if case.get("threshold"):
        return "threshold"
    if case.get("word_order"):
        return "word_order"
    return "other"


def build_split(report: dict) -> dict:
    rows = {
        name: [
            case
            for case in report["cases"]
            if case["field"] == name
            and not case["unmatched_prediction"]
            and not case["exact"]
            and case["overlap"] >= 0.5
        ]
        for name in FIELDS
    }
    fields = {}
    examples = {}
    for name, cases in rows.items():
        counts = Counter(split_label(case) for case in cases)
        fields[name] = {"located_wrong_text": len(cases), "split": dict(counts)}
        examples[name] = {}
        for label in ("ocr_wording", "extra_header", "both"):
            chosen = [case for case in cases if split_label(case) == label]
            chosen.sort(key=lambda case: -case["overlap"])
            examples[name][label] = [
                {
                    "document_id": case["document_id"],
                    "page": case["page"],
                    "overlap": case["overlap"],
                    "gold": case["gold"],
                    "prediction": case["prediction"],
                }
                for case in chosen[:3]
            ]
    return {
        "source": "address-text-audit-100.json",
        "validation_pages": report["validation_pages"],
        "documents": report["documents"],
        "threshold": report["threshold"],
        "grouping": report["grouping"],
        "checkpoint_updated": False,
        "fields": fields,
        "examples": examples,
    }


def render_split(split: dict) -> str:
    lines = [
        "# Address mismatch split",
        "",
        "Located addresses whose text is not exact. OCR wording, an extra header line, or both.",
        "The checkpoint, threshold, grouping, OCR, and annotations were not changed.",
        "",
        "| Field | Located, wrong text | OCR wording | Extra header | Both | Line break | Threshold | Word order |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in FIELDS:
        row = split["fields"][name]
        counts = row["split"]
        lines.append(
            "| {name} | {total} | {ocr} | {header} | {both} | {line} | {threshold} | {order} |".format(
                name=name,
                total=row["located_wrong_text"],
                ocr=counts.get("ocr_wording", 0),
                header=counts.get("extra_header", 0),
                both=counts.get("both", 0),
                line=counts.get("line_break", 0),
                threshold=counts.get("threshold", 0),
                order=counts.get("word_order", 0),
            )
        )
    lines.extend(["", "## Examples", ""])
    for name in FIELDS:
        lines.append(f"### {name}")
        lines.append("")
        for label in ("ocr_wording", "extra_header", "both"):
            samples = split["examples"][name][label]
            if not samples:
                continue
            lines.append(f"#### {label}")
            lines.append("")
            for sample in samples:
                lines.append(f"- `{sample['document_id']}` page {sample['page']} overlap {sample['overlap']}")
                lines.append(f"  - gold: {sample['gold']!r}")
                lines.append(f"  - prediction: {sample['prediction']!r}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "artifacts" / "diagnostics" / "address-text-audit-100.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "diagnostics" / "address-text-split-100.json",
    )
    args = parser.parse_args()
    report = json.loads(args.audit.read_text(encoding="utf-8"))
    split = build_split(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(split, indent=2), encoding="utf-8")
    summary = args.output.with_suffix(".md")
    summary.write_text(render_split(split), encoding="utf-8")
    print(summary.read_text(encoding="utf-8"))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
