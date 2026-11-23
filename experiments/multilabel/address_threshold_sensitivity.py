"""Offline address-threshold comparison on the labelled validation pages.

The frozen checkpoint, the 0.5 production threshold, and the extraction service
are not modified. Reader agreement uses the already saved crop reads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "experiments"))

from multilabel import FIELD_NAMES  # noqa: E402
from multilabel.address_reader_compare import text_inside  # noqa: E402
from multilabel.address_text_audit import (  # noqa: E402
    ADDRESS_FIELDS,
    classify_address_field,
    tokens,
    unmatched_predictions,
)
from multilabel.post_ocr import sequence_distance, stack_correction  # noqa: E402

CACHE = ROOT / "artifacts" / "diagnostics" / "address-threshold-word-probs.jsonl"
OUTPUT = ROOT / "artifacts" / "diagnostics" / "address-threshold-sensitivity.json"
SUMMARY = ROOT / "artifacts" / "diagnostics" / "address-threshold-sensitivity.md"
SHARED = (0.50, 0.45, 0.40, 0.35)
VENDOR = FIELD_NAMES.index("vendor_address")
CUSTOMER = FIELD_NAMES.index("customer_billing_address")


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
                "fieldtype": field["fieldtype"],
                "text": field.get("text") or "",
                "bbox": [min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])],
            }
        )
    return fields


def write_cache(checkpoint: Path, data_dir: Path, image_dir: Path) -> None:
    import torch

    from docintel.data.docile import load_docile_pages
    from multilabel.align import word_probabilities
    from multilabel.data import encode_document
    from multilabel.evaluate import load_multilabel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Reading checkpoint {checkpoint}", flush=True)
    model, processor, _payload = load_multilabel(checkpoint, device)
    docs = load_docile_pages(data_dir, "val", limit=100, image_dir=image_dir)
    print(f"Loaded pages {len(docs)}", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("w", encoding="utf-8") as handle:
        for count, doc in enumerate(docs, start=1):
            page = encode_document(doc, processor, max_length=512)
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
            seen = {word_id for word_id in page["word_ids"] if word_id is not None}
            words = []
            for index, text in enumerate(page["words"]):
                row = probabilities[index]
                words.append(
                    {
                        "text": text,
                        "box": [int(value) for value in page["boxes"][index]],
                        "vendor": float(row[VENDOR]),
                        "customer": float(row[CUSTOMER]),
                        "truncated": index not in seen,
                    }
                )
            doc_id, page_s = doc.document_id.rsplit("-p", 1)
            handle.write(
                json.dumps(
                    {
                        "document_id": doc_id,
                        "page": int(page_s),
                        "words": words,
                        "golds": _gold_fields(doc),
                    }
                )
                + "\n"
            )
            del page, batch, logits, probabilities
            if count % 20 == 0:
                print(f"scored {count}", flush=True)
    print(f"Wrote {CACHE}", flush=True)


def load_cache() -> list[dict]:
    pages = []
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            pages.append(json.loads(line))
    return pages


def load_readers(path: Path) -> dict[str, list[dict]]:
    pages: dict[str, list[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") == "meta":
            continue
        pages.setdefault(row["document_id"], []).extend(row.get("words") or [])
    return pages


def _probabilities(page: dict) -> list[list[float]]:
    rows = []
    for word in page["words"]:
        row = [0.0] * len(FIELD_NAMES)
        row[VENDOR] = word["vendor"]
        row[CUSTOMER] = word["customer"]
        rows.append(row)
    return rows


def _predict(page: dict, vendor: float, customer: float, *, anchor: float | None = None) -> list[dict]:
    from multilabel.group import group_fields

    cutoffs = [0.5] * len(FIELD_NAMES)
    cutoffs[VENDOR] = vendor
    cutoffs[CUSTOMER] = customer
    fields = group_fields(
        [word["text"] for word in page["words"]],
        [word["box"] for word in page["words"]],
        _probabilities(page),
        FIELD_NAMES,
        threshold=0.5,
        field_thresholds=cutoffs,
        page_index=page["page"],
        grouping="line-block-v2",
    )
    if anchor is not None:
        kept = []
        for field in fields:
            if field.label not in ADDRESS_FIELDS:
                continue
            column = VENDOR if field.label == "vendor_address" else CUSTOMER
            peak = max(page["words"][index]["vendor" if column == VENDOR else "customer"] for index in field.word_indexes)
            if peak >= anchor:
                kept.append(field)
        fields = kept
    return [
        {
            "document_id": page["document_id"],
            "fieldtype": field.label,
            "text": field.text,
            "bbox": field.box_relative,
            "score": field.confidence,
            "word_indexes": field.word_indexes,
            "page": field.page_index,
        }
        for field in fields
        if field.label in ADDRESS_FIELDS
    ]


def _column_words(page: dict, field: str) -> list[dict]:
    key = "vendor" if field == "vendor_address" else "customer"
    return [
        {
            "text": word["text"],
            "box": word["box"],
            "probability": word[key],
            "truncated": word["truncated"],
        }
        for word in page["words"]
    ]


def _reader_pair(page: dict, gold: dict, doctr: dict, ppocr: dict) -> list[str]:
    key = f"{page['document_id']}-p{page['page']}"
    return [
        text_inside(doctr.get(key, []), gold["bbox"]),
        text_inside(ppocr.get(key, []), gold["bbox"]),
    ]


def score_setting(pages: list[dict], vendor: float, customer: float, doctr: dict, ppocr: dict, *, anchor: float | None = None) -> dict:
    per_field = {
        name: {
            "gold": 0,
            "exact_raw": 0,
            "exact_stack": 0,
            "located": 0,
            "raw_fp_fields": 0,
            "stack_fp_fields": 0,
            "false_positive_words": 0,
            "unwanted_nearby_words": 0,
            "word_errors_raw": 0,
            "word_errors_stack": 0,
            "gold_words": 0,
        }
        for name in ADDRESS_FIELDS
    }
    recovered = []
    introduced = []
    for page in pages:
        predictions = _predict(page, vendor, customer, anchor=anchor)
        baseline = _predict(page, 0.50, 0.50) if (vendor, customer, anchor) != (0.50, 0.50, None) else predictions
        golds = [
            {
                "document_id": page["document_id"],
                "page": page["page"],
                "fieldtype": gold["fieldtype"],
                "text": gold["text"],
                "bbox": gold["bbox"],
            }
            for gold in page["golds"]
        ]
        cases = []
        for gold in golds:
            siblings = [
                item
                for item in golds
                if item is not gold and item["fieldtype"] == gold["fieldtype"] and item["page"] == gold["page"]
            ]
            cases.append(
                classify_address_field(
                    gold,
                    _column_words(page, gold["fieldtype"]),
                    predictions,
                    siblings,
                )
            )
        extras = unmatched_predictions(golds, predictions, cases)
        base_by_gold = {}
        if baseline is not predictions:
            base_cases = []
            for gold in golds:
                siblings = [
                    item
                    for item in golds
                    if item is not gold and item["fieldtype"] == gold["fieldtype"] and item["page"] == gold["page"]
                ]
                base_cases.append(
                    classify_address_field(gold, _column_words(page, gold["fieldtype"]), baseline, siblings)
                )
            for gold, case in zip(golds, base_cases):
                base_by_gold[(gold["fieldtype"], gold["text"], tuple(gold["bbox"]))] = case.get("prediction")
        for gold, case in zip(golds, cases):
            bucket = per_field[gold["fieldtype"]]
            bucket["gold"] += 1
            raw = case["prediction"] or ""
            located = case["overlap"] >= 0.5 and case["prediction"] is not None
            readers = _reader_pair(page, gold, doctr, ppocr) if located else []
            stacked = stack_correction(case["prediction"], readers) if located else ""
            exact_raw = bool(located and case["prediction"] == gold["text"])
            exact_stack = bool(located and stacked == gold["text"])
            bucket["exact_raw"] += int(exact_raw)
            bucket["exact_stack"] += int(exact_stack)
            bucket["located"] += int(located)
            if case["prediction"] is not None and not exact_raw:
                bucket["raw_fp_fields"] += 1
            if located and stacked and not exact_stack:
                bucket["stack_fp_fields"] += 1
            bucket["unwanted_nearby_words"] += len(case["unwanted_nearby"])
            bucket["false_positive_words"] += sum(
                1 for word in case["unwanted_nearby"] if not word.get("inside_gold_box")
            )
            gold_words = (gold["text"] or "").split()
            bucket["word_errors_raw"] += sequence_distance(raw.split(), gold_words)
            bucket["word_errors_stack"] += sequence_distance((stacked or "").split(), gold_words)
            bucket["gold_words"] += len(gold_words)
            if len(recovered) < 8 or len(introduced) < 8:
                previous = base_by_gold.get((gold["fieldtype"], gold["text"], tuple(gold["bbox"])))
                _collect_examples(recovered, introduced, gold, previous, case["prediction"])
        for extra in extras:
            bucket = per_field[extra["field"]]
            text = extra["prediction"] or ""
            bucket["raw_fp_fields"] += 1
            stacked = stack_correction(text, [])
            if stacked:
                bucket["stack_fp_fields"] += 1
            count = len(tokens(text))
            bucket["false_positive_words"] += count
            bucket["unwanted_nearby_words"] += count
    return _pack(vendor, customer, anchor, per_field, recovered, introduced)


def _collect_examples(recovered: list, introduced: list, gold: dict, previous: str | None, current: str | None) -> None:
    from multilabel.address_text_audit import norm_token

    before = {norm_token(token) for token in tokens(previous or "") if norm_token(token)}
    after = {norm_token(token) for token in tokens(current or "") if norm_token(token)}
    official = {norm_token(token) for token in tokens(gold["text"]) if norm_token(token)}
    gained = sorted(token for token in after - before if token in official)
    added = sorted(token for token in after - before if token not in official)
    if gained and len(recovered) < 8:
        recovered.append(
            {
                "document_id": gold["document_id"],
                "page": gold["page"],
                "field": gold["fieldtype"],
                "words": gained[:8],
                "gold": gold["text"],
                "baseline": previous,
                "prediction": current,
            }
        )
    if added and len(introduced) < 8:
        introduced.append(
            {
                "document_id": gold["document_id"],
                "page": gold["page"],
                "field": gold["fieldtype"],
                "words": added[:8],
                "gold": gold["text"],
                "baseline": previous,
                "prediction": current,
            }
        )


def _rates(tp: int, fp: int, gold: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / gold if gold else None
    return {"tp": tp, "fp": fp, "fn": gold - tp, "precision": precision, "recall": recall}


def _pack(vendor: float, customer: float, anchor: float | None, per_field: dict, recovered: list, introduced: list) -> dict:
    fields = {}
    for name, bucket in per_field.items():
        fields[name] = {
            "gold": bucket["gold"],
            "exact_raw": bucket["exact_raw"],
            "exact_stack": bucket["exact_stack"],
            "located": bucket["located"],
            "raw": _rates(bucket["exact_raw"], bucket["raw_fp_fields"], bucket["gold"]),
            "stack": _rates(bucket["exact_stack"], bucket["stack_fp_fields"], bucket["gold"]),
            "false_positive_words": bucket["false_positive_words"],
            "unwanted_nearby_words": bucket["unwanted_nearby_words"],
            "wer_raw": bucket["word_errors_raw"] / bucket["gold_words"] if bucket["gold_words"] else None,
            "wer_stack": bucket["word_errors_stack"] / bucket["gold_words"] if bucket["gold_words"] else None,
        }
    combined_gold = sum(fields[name]["gold"] for name in ADDRESS_FIELDS)
    combined_raw = sum(fields[name]["exact_raw"] for name in ADDRESS_FIELDS)
    combined_stack = sum(fields[name]["exact_stack"] for name in ADDRESS_FIELDS)
    raw_fp = sum(fields[name]["raw"]["fp"] for name in ADDRESS_FIELDS)
    stack_fp = sum(fields[name]["stack"]["fp"] for name in ADDRESS_FIELDS)
    return {
        "vendor_threshold": vendor,
        "customer_threshold": customer,
        "anchor": anchor,
        "fields": fields,
        "combined": {
            "gold": combined_gold,
            "exact_raw": combined_raw,
            "exact_stack": combined_stack,
            "raw": _rates(combined_raw, raw_fp, combined_gold),
            "stack": _rates(combined_stack, stack_fp, combined_gold),
            "false_positive_words": sum(fields[name]["false_positive_words"] for name in ADDRESS_FIELDS),
            "unwanted_nearby_words": sum(fields[name]["unwanted_nearby_words"] for name in ADDRESS_FIELDS),
        },
        "recovered_examples": recovered,
        "introduced_examples": introduced,
    }


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def render(report: dict) -> str:
    lines = [
        "# Address threshold sensitivity",
        "",
        "Labelled validation pages only. The frozen model, the production threshold 0.5,",
        "and the extraction service were not changed.",
        "",
        f"Pages: {report['pages']}. Gold addresses: {report['gold']}.",
        "Exact match requires the official string and a box overlap of at least 0.5.",
        "The stack is the existing character, boundary, and reader-agreement correction.",
        "False-positive words are span words outside the official box, plus every word in a span that overlaps no official address.",
        "Unwanted nearby words are span words that are not official address tokens.",
        "",
        "## Shared threshold",
        "",
        "| Threshold | Vendor exact | Customer exact | Combined raw | Combined after stack | Raw precision | Raw recall | Stack precision | Stack recall | False-positive words | Unwanted nearby | Vendor WER | Customer WER |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["shared"]:
        vendor = row["fields"]["vendor_address"]
        customer = row["fields"]["customer_billing_address"]
        lines.append(
            "| {cut:.2f} | {vex} | {cex} | {raw} | {stack} | {rp} | {rr} | {sp} | {sr} | {fp} | {near} | {vw} | {cw} |".format(
                cut=row["vendor_threshold"],
                vex=vendor["exact_raw"],
                cex=customer["exact_raw"],
                raw=row["combined"]["exact_raw"],
                stack=row["combined"]["exact_stack"],
                rp=_fmt(row["combined"]["raw"]["precision"]),
                rr=_fmt(row["combined"]["raw"]["recall"]),
                sp=_fmt(row["combined"]["stack"]["precision"]),
                sr=_fmt(row["combined"]["stack"]["recall"]),
                fp=row["combined"]["false_positive_words"],
                near=row["combined"]["unwanted_nearby_words"],
                vw=_fmt(vendor["wer_raw"]),
                cw=_fmt(customer["wer_raw"]),
            )
        )
    lines.extend(
        [
            "",
            "## Separate thresholds, combined exact after the existing stack",
            "",
            "| Vendor \\ Customer | " + " | ".join(f"{value:.2f}" for value in SHARED) + " |",
            "| --- | " + " | ".join("---:" for _ in SHARED) + " |",
        ]
    )
    for vendor in SHARED:
        cells = []
        for customer in SHARED:
            row = report["grid"][f"{vendor:.2f}/{customer:.2f}"]
            cells.append(f"{row['combined']['exact_stack']}/{row['combined']['gold']}")
        lines.append("| " + f"{vendor:.2f}" + " | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Context extension, offline only",
            "",
            "A span grouped at the lower cutoff is kept only when at least one of its words is already at or above 0.5.",
            "This rule was not added to production.",
            "",
            "| Extension | Combined raw | Combined after stack | Raw precision | False-positive words | Unwanted nearby |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["context"]:
        lines.append(
            "| {label} | {raw} | {stack} | {precision} | {fp} | {near} |".format(
                label=row["label"],
                raw=row["result"]["combined"]["exact_raw"],
                stack=row["result"]["combined"]["exact_stack"],
                precision=_fmt(row["result"]["combined"]["raw"]["precision"]),
                fp=row["result"]["combined"]["false_positive_words"],
                near=row["result"]["combined"]["unwanted_nearby_words"],
            )
        )
    lines.extend(["", "## Examples", ""])
    for title, key in (("Recovered official words", "recovered_examples"), ("Newly introduced words", "introduced_examples")):
        lines.append(f"### {title}")
        lines.append("")
        shown = False
        for row in report["shared"][1:] + [item["result"] for item in report["context"]]:
            samples = row[key][:2]
            if not samples:
                continue
            shown = True
            label = row.get("label") or f"threshold {row['vendor_threshold']:.2f}"
            for sample in samples:
                lines.append(
                    f"- {label} `{sample['document_id']}` page {sample['page']} {sample['field']}: {', '.join(sample['words'])}"
                )
        if not shown:
            lines.append("None recorded.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    from multilabel.checkpoint import assert_diagnostic_destination, assert_output_allowed

    assert_diagnostic_destination(OUTPUT)
    assert_diagnostic_destination(SUMMARY)
    assert_output_allowed(OUTPUT)
    if not CACHE.is_file():
        write_cache(
            ROOT / "artifacts" / "checkpoints" / "epoch-3",
            ROOT / "data" / "docile",
            ROOT / "data" / "docile" / "page_images",
        )
    pages = load_cache()
    doctr = load_readers(ROOT / "artifacts" / "diagnostics" / "address-reader-doctr-crops.jsonl")
    ppocr = load_readers(ROOT / "artifacts" / "diagnostics" / "address-reader-ppocrv5-crops.jsonl")
    shared = [score_setting(pages, value, value, doctr, ppocr) for value in SHARED]
    grid = {
        f"{vendor:.2f}/{customer:.2f}": score_setting(pages, vendor, customer, doctr, ppocr)
        for vendor in SHARED
        for customer in SHARED
    }
    context = []
    for cutoff in (0.45, 0.40, 0.35):
        result = score_setting(pages, cutoff, cutoff, doctr, ppocr, anchor=0.50)
        result["label"] = f"keep a 0.50 anchor, extend to {cutoff:.2f}"
        context.append({"label": result["label"], "result": result})
    gold = shared[0]["combined"]["gold"]
    report = {
        "pages": len(pages),
        "gold": gold,
        "production_changed": False,
        "baseline_reference": {"raw": "61/162", "stack": "72/162"},
        "shared": shared,
        "grid": grid,
        "context": context,
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    SUMMARY.write_text(render(report), encoding="utf-8")
    base = shared[0]
    print(
        "baseline raw",
        base["fields"]["vendor_address"]["exact_raw"],
        base["fields"]["customer_billing_address"]["exact_raw"],
        base["combined"]["exact_raw"],
        "stack",
        base["combined"]["exact_stack"],
        flush=True,
    )
    print(f"Wrote {OUTPUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
