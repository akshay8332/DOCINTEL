"""Diagnostics for boundaries, OCR text, and thresholds. Weights are not updated."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "experiments"))

from multilabel import FIELD_NAMES
from multilabel.data import encode_document, load_settings, prepare_documents, write_json
from multilabel.evaluate import load_multilabel, official_kile
from multilabel.group import GROUPING_LINE_BLOCK, MULTILINE_FIELDS, group_fields
from multilabel.ocr_text import correct_common_ocr

PARTY = (
    "vendor_name",
    "customer_billing_name",
    "vendor_address",
    "customer_billing_address",
)
AMOUNT = ("amount_due", "amount_total_gross")
NAME = ("vendor_name", "customer_billing_name")
ADDRESS = ("vendor_address", "customer_billing_address")
HEADER = re.compile(r"^(Advertiser|Agency|Bill|To|Station|FM|AM)$", re.IGNORECASE)
CALL_SIGN = re.compile(r"^[A-Z]{3,4}--[A-Z]{2}$")
CACHE = ROOT / "artifacts" / "diagnostics" / "multilabel-word-scores-100.jsonl"


def _contains(box: list[float], word_box: list[int]) -> bool:
    cx = (word_box[0] + word_box[2]) / 2000
    cy = (word_box[1] + word_box[3]) / 2000
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def _gold(doc) -> list[dict]:
    page = int(doc.document_id.rsplit("-p", 1)[1])
    payload = json.loads(Path(doc.annotation_path).read_text(encoding="utf-8"))
    fields = []
    for field in payload.get("field_extractions") or []:
        if int(field.get("page", 0)) != page or field.get("fieldtype") not in FIELD_NAMES:
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


def build_cache(settings: dict, limit: int) -> list[dict]:
    if CACHE.is_file() and sum(1 for _ in CACHE.open(encoding="utf-8")) >= limit:
        return [json.loads(line) for line in CACHE.read_text(encoding="utf-8").splitlines()[:limit]]
    import torch
    from multilabel.align import word_probabilities

    checkpoint = Path(settings["weights_checkpoint"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Caching scores from {checkpoint}", flush=True)
    model, processor, _payload = load_multilabel(checkpoint, device)
    docs = prepare_documents(Path(settings["data_dir"]), "val", limit)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with CACHE.open("w", encoding="utf-8") as handle:
        for index, doc in enumerate(docs, start=1):
            page = encode_document(doc, processor, max_length=int(settings["max_seq_length"]))
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
            row = {
                "document_id": doc.document_id,
                "words": page["words"],
                "boxes": page["boxes"],
                "probabilities": probabilities,
                "gold": _gold(doc),
            }
            handle.write(json.dumps(row) + "\n")
            rows.append(row)
            if index % 20 == 0:
                print(f"  cached {index}", flush=True)
    return rows


def _pages(rows: list[dict]) -> list[dict]:
    return [{"document_id": row["document_id"]} for row in rows]


def _held_out(rows: list[dict], original_pages: int) -> list[dict]:
    seen = {row["document_id"].rsplit("-p", 1)[0] for row in rows[:original_pages]}
    return [row for row in rows[original_pages:] if row["document_id"].rsplit("-p", 1)[0] not in seen]


def _predict(rows: list[dict], **kwargs) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        page = int(row["document_id"].rsplit("-p", 1)[1])
        for field in group_fields(
            row["words"],
            row["boxes"],
            row["probabilities"],
            FIELD_NAMES,
            page_index=page,
            grouping=GROUPING_LINE_BLOCK,
            **kwargs,
        ):
            grouped[row["document_id"].rsplit("-p", 1)[0]].append(field.to_docile_dict())
    return grouped


def _slim(official: dict) -> dict:
    if "iou_0.5" not in official:
        return official
    keep = (*PARTY, *AMOUNT, "date_due", "document_id", "date_issue")
    packed = {}
    for key in ("iou_1.0", "iou_0.5", "iou_1.0_exact_text"):
        block = official[key]
        packed[key] = {
            "nine": block["nine_trained_fields"],
            "fields": {name: block["per_field"][name] for name in keep},
        }
    return packed


def _score(settings: dict, rows: list[dict], predictions: dict) -> dict:
    if not rows:
        return {}
    return _slim(official_kile(settings, _pages(rows), predictions))


def _apply_text(predictions: dict, fn) -> dict:
    copied = {}
    for doc_id, fields in predictions.items():
        copied[doc_id] = [{**field, "text": fn(field["text"])} for field in fields]
    return copied


def _suppress_headers(row: dict) -> dict:
    probabilities = [list(row_probs) for row_probs in row["probabilities"]]
    party_columns = [FIELD_NAMES.index(name) for name in PARTY]
    for index, word in enumerate(row["words"]):
        if HEADER.fullmatch(word) or CALL_SIGN.fullmatch(word):
            for column in party_columns:
                probabilities[index][column] = 0.0
    return {**row, "probabilities": probabilities}


def _outside_party_words(rows: list[dict]) -> dict:
    distances = []
    examples = []
    header_hits = Counter()
    positive = 0
    outside = 0
    for row in rows:
        for column, name in enumerate(FIELD_NAMES):
            if name not in PARTY:
                continue
            gold_boxes = [field["bbox"] for field in row["gold"] if field["fieldtype"] == name]
            for index, word in enumerate(row["words"]):
                if row["probabilities"][index][column] < 0.5:
                    continue
                positive += 1
                box = row["boxes"][index]
                if any(_contains(gold, box) for gold in gold_boxes):
                    continue
                outside += 1
                center_y = (box[1] + box[3]) / 2000
                above = [gold[1] - center_y for gold in gold_boxes if gold[1] > center_y]
                gap = min(above) if above else None
                if gap is not None:
                    distances.append(gap)
                if HEADER.fullmatch(word) or CALL_SIGN.fullmatch(word):
                    header_hits[word] += 1
                if len(examples) < 12:
                    examples.append(
                        {
                            "document_id": row["document_id"],
                            "field": name,
                            "text": word,
                            "probability": round(row["probabilities"][index][column], 3),
                            "gap_above_gold": None if gap is None else round(gap, 4),
                        }
                    )
    distances.sort()
    return {
        "positive_party_words": positive,
        "outside_gold_box": outside,
        "outside_fraction": outside / positive if positive else 0.0,
        "median_gap_above_gold": distances[len(distances) // 2] if distances else None,
        "header_or_call_sign": dict(header_hits),
        "examples": examples,
    }


def _missing_ocr(rows: list[dict]) -> dict:
    missing = Counter()
    compared = Counter()
    examples = []
    for row in rows:
        for field in row["gold"]:
            if field["fieldtype"] not in (*PARTY, *AMOUNT, "date_issue", "document_id"):
                continue
            inside = [
                word
                for word, box in zip(row["words"], row["boxes"])
                if _contains(field["bbox"], box)
            ]
            for token in field["text"].replace("\n", " ").split():
                compared[field["fieldtype"]] += 1
                if token not in inside:
                    missing[field["fieldtype"]] += 1
                    if len(examples) < 12:
                        examples.append(
                            {
                                "document_id": row["document_id"],
                                "field": field["fieldtype"],
                                "token": token,
                                "ocr_inside_box": inside[:12],
                            }
                        )
    return {
        "gold_tokens": dict(compared),
        "tokens_absent_from_ocr_box": dict(missing),
        "examples": examples,
    }


def _gold_header_count(rows: list[dict]) -> dict:
    counts = Counter()
    for row in rows:
        for field in row["gold"]:
            if field["fieldtype"] not in PARTY:
                continue
            for token in field["text"].replace("\n", " ").split():
                if HEADER.fullmatch(token) or CALL_SIGN.fullmatch(token):
                    counts[token] += 1
    return dict(counts)


def _metrics_line(score: dict) -> dict:
    block = score.get("iou_0.5", {})
    nine = block.get("nine", {})
    return {
        "f1": nine.get("f1"),
        "precision": nine.get("precision"),
        "recall": nine.get("recall"),
        "TP": nine.get("TP"),
        "FP": nine.get("FP"),
        "FN": nine.get("FN"),
        "fields": {
            name: block.get("fields", {}).get(name)
            for name in (*PARTY, *AMOUNT)
        },
    }


def main() -> int:
    settings = load_settings(ROOT / "configs" / "multilabel-200.yaml")
    rows = build_cache(settings, 100)
    original = int(settings["eval_limit"])
    held = _held_out(rows, original)
    print(f"Cached {len(rows)} pages, held-out {len(held)}", flush=True)
    baseline = _predict(held)
    report = {
        "checkpoint_updated": False,
        "weights": str(settings["weights_checkpoint"]),
        "grouping": GROUPING_LINE_BLOCK,
        "threshold_baseline": 0.5,
        "held_out_pages": len(held),
        "full_training_started": False,
        "party_boundary": {
            "outside_words": _outside_party_words(held),
            "header_tokens_inside_gold": _gold_header_count(held),
            "multiline_fields": sorted(MULTILINE_FIELDS),
        },
        "ocr": {
            "original_files_modified": False,
            "missing_text": _missing_ocr(held),
        },
        "threshold_experiments": [],
        "boundary_experiments": [],
    }
    report["ocr"]["character_correction"] = {
        "baseline": _metrics_line(_score(settings, held, baseline)),
        "corrected_text": _metrics_line(_score(settings, held, _apply_text(baseline, correct_common_ocr))),
    }
    print("OCR comparison scored", flush=True)
    groups = {"names": NAME, "addresses": ADDRESS, "amounts": AMOUNT}
    for group_name, members in groups.items():
        for cutoff in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            thresholds = []
            for name in FIELD_NAMES:
                thresholds.append(cutoff if name in members else 0.5)
            scored = _score(settings, held, _predict(held, field_thresholds=thresholds))
            report["threshold_experiments"].append(
                {"group": group_name, "threshold": cutoff, "iou_0.5": _metrics_line(scored)}
            )
            print(f"  threshold {group_name} {cutoff}", flush=True)
    for gap in (40, 20, 10):
        scored = _score(settings, held, _predict(held, block_gap=gap))
        report["boundary_experiments"].append({"block_gap": gap, "iou_0.5": _metrics_line(scored)})
    suppressed = [_suppress_headers(row) for row in held]
    report["boundary_experiments"].append(
        {
            "header_suppression": True,
            "block_gap": 40,
            "iou_0.5": _metrics_line(_score(settings, held, _predict(suppressed))),
        }
    )
    destination = ROOT / "artifacts" / "diagnostics" / "phase4-diagnostics.json"
    write_json(destination, report)
    print(f"Wrote {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
