"""Diagnostics for the 200-page invoice LayoutLMv3 run.

Reads the same train/val pages as training, then scores the saved checkpoint.
Does not modify the checkpoint.
"""

from __future__ import annotations

import json
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))

from docintel.data.docile import INVOICE_BIO_LABELS, INVOICE_FIELDS, load_docile_pages
from docintel.data.transforms import document_to_example, encode_example_for_layoutlm
from docintel.models.layoutlm import load_layoutlm_for_token_classification
from docintel.models.metrics import align_predictions, entity_f1_report

DATA = ROOT / "data" / "docile"
CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "invoice-layoutlm-200"
OUT = ROOT / "artifacts" / "diagnostics" / "invoice-200.json"
DOCILE_SRC = Path(r"C:\Users\DELL\Downloads\docile-main\docile-main")
KILE_FIELDTYPES = [
    "account_num",
    "amount_due",
    "amount_paid",
    "amount_total_gross",
    "amount_total_net",
    "amount_total_tax",
    "bank_num",
    "bic",
    "currency_code_amount_due",
    "customer_billing_address",
    "customer_billing_name",
    "customer_delivery_address",
    "customer_delivery_name",
    "customer_id",
    "customer_order_id",
    "customer_other_address",
    "customer_other_name",
    "customer_registration_id",
    "customer_tax_id",
    "date_due",
    "date_issue",
    "document_id",
    "iban",
    "order_id",
    "payment_reference",
    "payment_terms",
    "tax_detail_gross",
    "tax_detail_net",
    "tax_detail_rate",
    "tax_detail_tax",
    "vendor_address",
    "vendor_email",
    "vendor_name",
    "vendor_order_id",
    "vendor_registration_id",
    "vendor_tax_id",
]


def _center(box: list[int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2000.0, (box[1] + box[3]) / 2000.0)


def _contains(bbox: list[float], cx: float, cy: float) -> bool:
    x0, y0, x1, y1 = bbox
    return min(x0, x1) <= cx <= max(x0, x1) and min(y0, y1) <= cy <= max(y0, y1)


def _iou(a: list[float], b: list[float]) -> float:
    x0 = max(min(a[0], a[2]), min(b[0], b[2]))
    y0 = max(min(a[1], a[3]), min(b[1], b[3]))
    x1 = min(max(a[0], a[2]), max(b[0], b[2]))
    y1 = min(max(a[1], a[3]), max(b[1], b[3]))
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = abs(a[2] - a[0]) * abs(a[3] - a[1])
    area_b = abs(b[2] - b[0]) * abs(b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _rel(box: list[int] | None) -> list[float] | None:
    if box is None:
        return None
    return [round(v / 1000.0, 4) for v in box]


def _norm_text(text: str) -> str:
    return " ".join(text.replace("&amp;", "&").split()).casefold()


def _gold_spans(example) -> list[dict]:
    spans = []
    i = 0
    words = example.words
    labels = example.word_labels
    boxes = example.boxes
    while i < len(labels):
        lab = labels[i]
        if not lab.startswith("B-"):
            i += 1
            continue
        entity = lab[2:]
        j = i + 1
        while j < len(labels) and labels[j] == f"I-{entity}":
            j += 1
        span_boxes = boxes[i:j]
        box = [
            min(b[0] for b in span_boxes),
            min(b[1] for b in span_boxes),
            max(b[2] for b in span_boxes),
            max(b[3] for b in span_boxes),
        ]
        spans.append(
            {
                "field": entity,
                "text": " ".join(words[i:j]),
                "box": box,
            }
        )
        i = j
    return spans


def _spread(items: list[dict], n: int = 10) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        buckets[item["field"]].append(item)
    chosen: list[dict] = []
    types = list(buckets)
    while len(chosen) < n and any(buckets.values()):
        for name in types:
            if buckets[name] and len(chosen) < n:
                chosen.append(buckets[name].pop(0))
    return chosen


def _as_list(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        return value[0]
    return list(value)


def alignment_encoding(processor, example):
    """Tokenize the same way LayoutLMv3 does, keeping word ids."""
    return processor.tokenizer(
        example.words,
        boxes=example.boxes,
        word_labels=example.label_ids,
        truncation=True,
        padding="max_length",
        max_length=512,
    )


def check_tokenization(example, encoding, wrapped_labels) -> dict:
    labels = [int(v) for v in _as_list(encoding["labels"])]
    word_ids = list(encoding.word_ids())
    wrapped = [int(v) for v in _as_list(wrapped_labels)]
    label_mismatch_vs_wrapper = sum(a != b for a, b in zip(labels, wrapped)) + abs(
        len(labels) - len(wrapped)
    )
    mismatch = 0
    special_not_ignored = 0
    continuation_labeled = 0
    continuation_ignored = 0
    first_subwords = 0
    supervised = 0
    o_tokens = 0
    seen: set[int] = set()
    for lab, wid in zip(labels, word_ids):
        if wid is None:
            if lab != -100:
                special_not_ignored += 1
            continue
        if wid not in seen:
            seen.add(wid)
            first_subwords += 1
            expected = example.label_ids[wid] if wid < len(example.label_ids) else None
            if lab != expected:
                mismatch += 1
            if lab != -100:
                supervised += 1
                if lab == 0:
                    o_tokens += 1
        else:
            if lab == -100:
                continuation_ignored += 1
            else:
                continuation_labeled += 1
                if lab != example.label_ids[wid]:
                    mismatch += 1
                supervised += 1
                if lab == 0:
                    o_tokens += 1
    dropped = len(example.words) - len(seen)
    dropped_labeled = sum(
        1
        for i, lab in enumerate(example.word_labels)
        if i not in seen and lab != "O"
    )
    return {
        "tokens": len(labels),
        "first_subwords": first_subwords,
        "words": len(example.words),
        "words_dropped": dropped,
        "labeled_words_dropped": dropped_labeled,
        "first_or_content_mismatches": mismatch,
        "special_not_ignored": special_not_ignored,
        "continuation_ignored": continuation_ignored,
        "continuation_labeled": continuation_labeled,
        "supervised_tokens": supervised,
        "o_tokens": o_tokens,
        "wrapper_label_mismatches": label_mismatch_vs_wrapper,
    }


def train_statistics(docs) -> dict:
    ann_counts: Counter[str] = Counter()
    word_counts: Counter[str] = Counter()
    entity_counts: Counter[str] = Counter()
    covered = Counter()
    uncovered = Counter()
    lost_to_smaller = Counter()
    ious: dict[str, list[float]] = defaultdict(list)
    uncovered_examples = []
    low_iou_examples = []
    total_words = 0
    o_words = 0

    for doc in docs:
        page_index = int(doc.document_id.rsplit("-p", 1)[1])
        annotation = json.loads(Path(doc.annotation_path).read_text(encoding="utf-8"))
        words = doc.pages[0].words
        centers = [_center(w.box) for w in words]
        total_words += len(words)
        for word in words:
            label = word.label or "O"
            word_counts[label] += 1
            if label == "O":
                o_words += 1
            elif label.startswith("B-"):
                entity_counts[label[2:]] += 1

        for field in annotation.get("field_extractions") or []:
            if int(field.get("page", 0)) != page_index:
                continue
            ftype = str(field.get("fieldtype", ""))
            bbox = field.get("bbox")
            if not ftype or not bbox or len(bbox) != 4:
                continue
            ann_counts[ftype] += 1
            if ftype not in INVOICE_FIELDS:
                continue
            box = [float(v) for v in bbox]
            inside = [
                w
                for w, (cx, cy) in zip(words, centers)
                if _contains(box, cx, cy)
            ]
            own_label = any(
                (w.label or "").endswith(ftype) and (w.label or "") != "O" for w in inside
            )
            if not inside:
                uncovered[ftype] += 1
                if len(uncovered_examples) < 8:
                    uncovered_examples.append(
                        {
                            "document_id": doc.document_id,
                            "field": ftype,
                            "text": field.get("text"),
                            "bbox": [round(float(v), 4) for v in box],
                        }
                    )
                continue
            covered[ftype] += 1
            if not own_label:
                lost_to_smaller[ftype] += 1
            union = [
                min(w.box[0] for w in inside) / 1000.0,
                min(w.box[1] for w in inside) / 1000.0,
                max(w.box[2] for w in inside) / 1000.0,
                max(w.box[3] for w in inside) / 1000.0,
            ]
            score = _iou(union, box)
            ious[ftype].append(score)
            if score < 0.5 and len(low_iou_examples) < 8:
                low_iou_examples.append(
                    {
                        "document_id": doc.document_id,
                        "field": ftype,
                        "annotation_text": field.get("text"),
                        "ocr_text": " ".join(w.text for w in inside),
                        "iou": round(score, 4),
                        "words": len(inside),
                    }
                )

    def _median(values: list[float]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return round(ordered[mid], 4)
        return round((ordered[mid - 1] + ordered[mid]) / 2, 4)

    per_field = []
    for name in INVOICE_FIELDS:
        per_field.append(
            {
                "field": name,
                "annotation_instances": ann_counts[name],
                "word_entities": entity_counts[name],
                "words": word_counts[f"B-{name}"] + word_counts[f"I-{name}"],
                "covered_by_ocr_center": covered[name],
                "no_ocr_word_center": uncovered[name],
                "covered_but_label_lost_to_smaller_box": lost_to_smaller[name],
                "median_iou": _median(ious[name]),
            }
        )
    absent_trained = [name for name in INVOICE_FIELDS if ann_counts[name] == 0]
    absent_kile = [name for name in KILE_FIELDTYPES if ann_counts[name] == 0]
    present_untrained = [
        {"field": name, "annotation_instances": ann_counts[name]}
        for name in KILE_FIELDTYPES
        if name not in INVOICE_FIELDS and ann_counts[name] > 0
    ]
    return {
        "pages": len(docs),
        "words": total_words,
        "o_words": o_words,
        "o_word_percent": round(100.0 * o_words / total_words, 2) if total_words else None,
        "per_field": per_field,
        "trained_fields_never_present": absent_trained,
        "kile_fields_never_present": absent_kile,
        "other_kile_fields_present_but_not_trained": present_untrained,
        "uncovered_examples": uncovered_examples,
        "low_iou_examples": low_iou_examples,
    }


def _batched(value):
    import torch

    if hasattr(value, "dim"):
        tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
    else:
        tensor = torch.tensor(value)
    if tensor.dim() == 1 or (tensor.dim() == 3 and tensor.shape[0] in (1, 3)):
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 2 and tensor.shape[-1] == 4:
        tensor = tensor.unsqueeze(0)
    return tensor


def predict_page(bundle, example):
    import torch
    from PIL import Image

    image = Image.open(example.image_path).convert("RGB")
    encoding = bundle.processor(
        image,
        example.words,
        boxes=example.boxes,
        word_labels=example.label_ids,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    data = dict(encoding)
    pixels = data.get("pixel_values")
    if isinstance(pixels, (list, tuple)) and len(pixels) == 1:
        data["pixel_values"] = pixels[0]
    tensor_inputs = {
        key: _batched(data[key])
        for key in ("input_ids", "attention_mask", "bbox", "token_type_ids", "pixel_values")
        if key in data
    }
    with torch.inference_mode():
        logits = bundle.model(**tensor_inputs).logits[0]
    probs = torch.softmax(logits, dim=-1)
    pred_ids = logits.argmax(dim=-1).tolist()
    conf_vals = probs.max(dim=-1).values.tolist()
    word_ids = list(encoding.word_ids())
    id2 = {i: name for i, name in enumerate(bundle.label_list)}
    word_labels = ["O"] * len(example.words)
    word_conf = [0.0] * len(example.words)
    seen: set[int] = set()
    for pos, wid in enumerate(word_ids):
        if wid is None or wid in seen or wid >= len(example.words):
            continue
        seen.add(wid)
        word_labels[wid] = id2.get(pred_ids[pos], "O")
        word_conf[wid] = float(conf_vals[pos])
    from docintel.models.layoutlm import _spans_from_bio

    fields = _spans_from_bio(example.words, word_labels, word_conf, example.boxes)
    labels = [int(v) for v in _as_list(encoding["labels"])]
    return encoding, pred_ids, labels, fields


def main() -> None:
    print("Loading 200 train pages", flush=True)
    train_docs = load_docile_pages(DATA, "train", limit=200)
    print(f"Loaded train pages: {len(train_docs)}", flush=True)
    stats = train_statistics(train_docs)

    print("Loading 20 val pages", flush=True)
    val_docs = load_docile_pages(DATA, "val", limit=20)
    print(f"Loaded val pages: {len(val_docs)}", flush=True)

    print("Loading checkpoint", flush=True)
    bundle = load_layoutlm_for_token_classification(
        str(CHECKPOINT),
        label_list=INVOICE_BIO_LABELS,
        prefer_v3=True,
    )
    bundle.model.eval()
    id2label = {i: name for i, name in enumerate(INVOICE_BIO_LABELS)}

    token_totals = Counter()
    token_examples = []
    print("Checking BIO alignment on train pages", flush=True)
    for n, doc in enumerate(train_docs, start=1):
        example = document_to_example(doc, label_list=INVOICE_BIO_LABELS)
        encoding = encode_example_for_layoutlm(
            example, bundle.processor, max_length=512
        )
        aligned = alignment_encoding(bundle.processor, example)
        report = check_tokenization(example, aligned, encoding["labels"])
        token_totals.update(report)
        if len(token_examples) < 3 and report["words_dropped"]:
            token_examples.append(
                {
                    "document_id": example.document_id,
                    "words": report["words"],
                    "words_dropped": report["words_dropped"],
                }
            )
        if n % 50 == 0:
            print(f"  tokenized {n}", flush=True)

    supervised = token_totals["supervised_tokens"]
    token_summary = {
        "pages": len(train_docs),
        "supervised_tokens": supervised,
        "o_tokens": token_totals["o_tokens"],
        "o_token_percent": round(100.0 * token_totals["o_tokens"] / supervised, 2)
        if supervised
        else None,
        "first_or_content_mismatches": token_totals["first_or_content_mismatches"],
        "wrapper_label_mismatches": token_totals["wrapper_label_mismatches"],
        "special_not_ignored": token_totals["special_not_ignored"],
        "continuation_ignored": token_totals["continuation_ignored"],
        "continuation_labeled": token_totals["continuation_labeled"],
        "words": token_totals["words"],
        "words_dropped_by_truncation": token_totals["words_dropped"],
        "labeled_words_dropped_by_truncation": token_totals["labeled_words_dropped"],
        "truncation_examples": token_examples,
        "rule": (
            "A token label is correct when it is -100 or equal to its word label. "
            "The first subword carries the word label; later subwords are -100 "
            "unless the processor labels them too."
        ),
    }

    print("Predicting 20 val pages", flush=True)
    gold_token_rows = []
    pred_token_rows = []
    correct = []
    missed = []
    false_positives = 0
    pred_box_samples = []
    doc_predictions: dict[str, list] = defaultdict(list)
    pages_by_doc: dict[str, list[int]] = defaultdict(list)

    for doc in val_docs:
        example = document_to_example(doc, label_list=INVOICE_BIO_LABELS)
        _encoding, pred_ids, labels, result_fields = predict_page(bundle, example)
        true_rows, pred_rows = align_predictions([pred_ids], [labels], id2label)
        gold_token_rows.extend(true_rows)
        pred_token_rows.extend(pred_rows)

        gold = _gold_spans(example)
        preds = [
            {
                "field": field.label,
                "text": field.text,
                "box": field.box,
                "confidence": round(field.confidence, 4),
            }
            for field in result_fields
        ]
        used = set()
        for g in gold:
            best_i = None
            best_iou = 0.0
            for i, pred in enumerate(preds):
                if i in used or pred["field"] != g["field"] or pred["box"] is None:
                    continue
                score = _iou(g["box"], pred["box"])
                if score > best_iou:
                    best_iou = score
                    best_i = i
            text_hit = any(
                pred["field"] == g["field"] and _norm_text(pred["text"]) == _norm_text(g["text"])
                for pred in preds
            )
            if best_i is not None and best_iou >= 0.5:
                used.add(best_i)
                pred = preds[best_i]
                correct.append(
                    {
                        "document_id": example.document_id,
                        "field": g["field"],
                        "gold_text": g["text"],
                        "pred_text": pred["text"],
                        "confidence": pred["confidence"],
                        "iou": round(best_iou, 4),
                        "text_match": _norm_text(pred["text"]) == _norm_text(g["text"]),
                        "gold_box_0_1000": g["box"],
                        "pred_box_0_1000": pred["box"],
                        "pred_box_relative": _rel(pred["box"]),
                    }
                )
            else:
                missed.append(
                    {
                        "document_id": example.document_id,
                        "field": g["field"],
                        "gold_text": g["text"],
                        "gold_box_relative": _rel(g["box"]),
                        "best_iou": round(best_iou, 4),
                        "text_match_elsewhere": text_hit,
                    }
                )
        false_positives += len(preds) - len(used)

        doc_id, page_s = example.document_id.rsplit("-p", 1)
        page_index = int(page_s)
        pages_by_doc[doc_id].append(page_index)
        for pred in preds:
            if pred["box"] is None:
                continue
            relative = _rel(pred["box"])
            doc_predictions[doc_id].append(
                {
                    "bbox": relative,
                    "page": page_index,
                    "score": pred["confidence"],
                    "text": pred["text"],
                    "fieldtype": pred["field"],
                    "raw_box_0_1000": pred["box"],
                }
            )
            if len(pred_box_samples) < 3:
                pred_box_samples.append(
                    {
                        "document_id": example.document_id,
                        "field": pred["field"],
                        "text": pred["text"],
                        "box_returned_by_model": pred["box"],
                        "docile_relative_box": relative,
                    }
                )

    custom = entity_f1_report(gold_token_rows, pred_token_rows)
    custom.pop("detail", None)

    box_coords = [
        coord
        for sample in pred_box_samples
        for coord in sample["box_returned_by_model"]
    ]
    format_check = {
        "pipeline_converts_to_docile_relative": False,
        "stored_coordinate_space": "0-1000 LayoutLM integers, union of predicted word boxes",
        "docile_coordinate_space": "0-1 [x0, y0, x1, y1] plus a 0-based page index",
        "sample_max_coordinate": max(box_coords) if box_coords else None,
        "samples": pred_box_samples,
        "pages_scored": {doc_id: sorted(pages) for doc_id, pages in pages_by_doc.items()},
    }

    official = run_official(pages_by_doc, doc_predictions)

    payload = {
        "field_counts": stats,
        "tokenization": token_summary,
        "val_custom_seqeval": custom,
        "val_span_match_iou_0.5": {
            "correct": len(correct),
            "missed": len(missed),
            "false_positives": false_positives,
        },
        "correct_fields": _spread(correct, 10),
        "missed_fields": _spread(missed, 10),
        "bbox_format": format_check,
        "official_evaluator": official,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}", flush=True)


def run_official(pages_by_doc: dict, doc_predictions: dict) -> dict:
    if not DOCILE_SRC.is_dir():
        return {"available": False, "reason": f"DocILE source not found at {DOCILE_SRC}"}
    sys.path.insert(0, str(DOCILE_SRC))
    try:
        from docile.dataset import BBox, Dataset, Field
        from docile.dataset.cached_object import CachingConfig
        from docile.evaluation.evaluate import evaluate_dataset
    except Exception as exc:
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    docids = list(pages_by_doc)
    predictions = {}
    for doc_id in docids:
        fields = []
        for pred in doc_predictions.get(doc_id, []):
            fields.append(
                Field(
                    bbox=BBox(*pred["bbox"]),
                    page=pred["page"],
                    score=pred["score"],
                    text=pred["text"],
                    fieldtype=pred["fieldtype"],
                )
            )
        predictions[doc_id] = fields
    try:
        dataset = Dataset(
            "invoice-pilot",
            DATA,
            load_annotations=True,
            load_ocr=True,
            cache_images=CachingConfig.OFF,
            docids=docids,
        )
        result_strict = evaluate_dataset(dataset, predictions, {}, iou_threshold=1.0)
        result_debug = evaluate_dataset(dataset, predictions, {}, iou_threshold=0.5)
    except Exception as exc:
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    def pack(result, same_text: bool = False) -> dict:
        overall = result.get_metrics("kile", same_text=same_text)
        per_field = {
            name: result.get_metrics("kile", same_text=same_text, fieldtype=name)
            for name in INVOICE_FIELDS
        }
        tp = sum(row["TP"] for row in per_field.values())
        fp = sum(row["FP"] for row in per_field.values())
        fn = sum(row["FN"] for row in per_field.values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return {
            "all_kile_fields_on_these_documents": {
                key: overall[key] for key in ("AP", "f1", "precision", "recall", "TP", "FP", "FN")
            },
            "nine_trained_fields_micro": {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "TP": tp,
                "FP": fp,
                "FN": fn,
            },
            "per_field": {
                name: {key: row[key] for key in ("AP", "f1", "precision", "recall", "TP", "FP", "FN")}
                for name, row in per_field.items()
            },
        }

    return {
        "available": True,
        "documents": len(docids),
        "predictions": sum(len(v) for v in predictions.values()),
        "iou_1.0": pack(result_strict),
        "iou_0.5": pack(result_debug),
        "iou_1.0_same_text": pack(result_strict, same_text=True)["nine_trained_fields_micro"],
    }


if __name__ == "__main__":
    main()
