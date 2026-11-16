"""Rebuild val spans with seqeval's chunk rules and score them with DocILE.

The checkpoint predicts I- tags without a B- tag. seqeval still starts an entity
on that I- tag. The project's span builder does not, so it returns no fields.
"""

from __future__ import annotations

import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "scripts"))

from seqeval.metrics.sequence_labeling import end_of_chunk, start_of_chunk

from diagnose_invoice import (  # noqa: E402
    DATA,
    DOCILE_SRC,
    INVOICE_BIO_LABELS,
    INVOICE_FIELDS,
    OUT,
    _gold_spans,
    _iou,
    _norm_text,
    _rel,
    _spread,
    predict_page,
)
from docintel.data.docile import load_docile_pages
from docintel.data.transforms import document_to_example
from docintel.models.layoutlm import load_layoutlm_for_token_classification
from docintel.models.metrics import align_predictions, entity_f1_report


def _tag_parts(label: str) -> tuple[str, str]:
    if not label or label == "O" or "-" not in label:
        return "O", ""
    tag, kind = label.split("-", 1)
    return tag, kind


def spans_like_seqeval(words, labels, confidences, boxes) -> list[dict]:
    spans = []
    prev_tag, prev_type = "O", ""
    begin = 0
    sequence = list(labels) + ["O"]
    for i, lab in enumerate(sequence):
        tag, kind = _tag_parts(lab)
        if end_of_chunk(prev_tag, tag, prev_type, kind):
            span_boxes = boxes[begin:i]
            spans.append(
                {
                    "field": prev_type,
                    "text": " ".join(words[begin:i]),
                    "box": [
                        min(b[0] for b in span_boxes),
                        min(b[1] for b in span_boxes),
                        max(b[2] for b in span_boxes),
                        max(b[3] for b in span_boxes),
                    ],
                    "confidence": round(sum(confidences[begin:i]) / max(i - begin, 1), 4),
                    "started_with_b": labels[begin].startswith("B-"),
                }
            )
        if start_of_chunk(prev_tag, tag, prev_type, kind):
            begin = i
        prev_tag, prev_type = tag, kind
    return spans


def word_predictions(encoding, pred_ids, example) -> tuple[list[str], list[float]]:
    id2 = {i: name for i, name in enumerate(INVOICE_BIO_LABELS)}
    word_ids = list(encoding.word_ids())
    labels = ["O"] * len(example.words)
    conf_source = pred_ids  # confidence filled by caller via parallel list
    del conf_source
    seen: set[int] = set()
    # Confidence is attached by the caller through encoding side channel.
    confs = getattr(encoding, "_word_conf_debug", None)
    word_conf = [0.0] * len(example.words)
    for pos, wid in enumerate(word_ids):
        if wid is None or wid in seen or wid >= len(example.words):
            continue
        seen.add(wid)
        labels[wid] = id2.get(int(pred_ids[pos]), "O")
        if confs is not None:
            word_conf[wid] = float(confs[pos])
    return labels, word_conf


def main() -> None:
    print("Loading checkpoint and 20 val pages", flush=True)
    bundle = load_layoutlm_for_token_classification(
        str(ROOT / "artifacts" / "checkpoints" / "invoice-layoutlm-200"),
        label_list=INVOICE_BIO_LABELS,
        prefer_v3=True,
    )
    bundle.model.eval()
    val_docs = load_docile_pages(DATA, "val", limit=20)
    id2label = {i: name for i, name in enumerate(INVOICE_BIO_LABELS)}

    import torch

    correct = []
    missed = []
    false_positives = 0
    pred_box_samples = []
    doc_predictions = defaultdict(list)
    pages_by_doc = defaultdict(list)
    tag_counts = defaultdict(int)
    production_spans = 0
    seqeval_spans = 0
    gold_token_rows = []
    pred_token_rows = []

    for doc in val_docs:
        example = document_to_example(doc, label_list=INVOICE_BIO_LABELS)
        encoding, pred_ids, labels, production = predict_page(bundle, example)
        true_rows, pred_rows = align_predictions([pred_ids], [labels], id2label)
        gold_token_rows.extend(true_rows)
        pred_token_rows.extend(pred_rows)
        production_spans += len(production)

        probs = None
        # Recompute confidence from a second pass would be wasteful. predict_page
        # already mapped the first-subword label; recover confidence from logits
        # by reading the stored prediction ids only. Use 1.0 placeholder only if
        # we cannot recover softmax. We re-read confidences from the encoding
        # forward inside predict_page by reconstructing word labels here.
        word_ids = list(encoding.word_ids())
        # Confidence: run softmax was not returned. Approximate with 0 and fill
        # from a cheap hook: predict_page does not return confidences of tokens.
        # Repeat the mapping using uniform confidence, then replace from logits
        # if pixel inputs are still available. Instead, store token confidence
        # by calling the model once more is too slow. Use the production field
        # confidences when a word was labeled, else read nothing.
        # We need real confidences for DocILE AP. Compute them from pred_ids
        # alignment only if we softmax again. Do it: the encoding is in memory.
        data = dict(encoding)
        pixels = data.get("pixel_values")
        if isinstance(pixels, (list, tuple)) and len(pixels) == 1:
            data["pixel_values"] = pixels[0]
        from diagnose_invoice import _batched

        tensor_inputs = {
            key: _batched(data[key])
            for key in ("input_ids", "attention_mask", "bbox", "token_type_ids", "pixel_values")
            if key in data
        }
        with torch.inference_mode():
            logits = bundle.model(**tensor_inputs).logits[0]
        conf_vals = torch.softmax(logits, dim=-1).max(dim=-1).values.tolist()
        word_labels = ["O"] * len(example.words)
        word_conf = [0.0] * len(example.words)
        seen: set[int] = set()
        for pos, wid in enumerate(word_ids):
            if wid is None or wid in seen or wid >= len(example.words):
                continue
            seen.add(wid)
            word_labels[wid] = id2label.get(int(pred_ids[pos]), "O")
            word_conf[wid] = float(conf_vals[pos])
            if labels[pos] != -100:
                tag_counts[word_labels[wid]] += 1

        preds = spans_like_seqeval(
            example.words, word_labels, word_conf, example.boxes
        )
        seqeval_spans += len(preds)
        gold = _gold_spans(example)
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
                        "started_with_b": pred["started_with_b"],
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
                    }
                )
        false_positives += len(preds) - len(used)

        doc_id, page_s = example.document_id.rsplit("-p", 1)
        page_index = int(page_s)
        pages_by_doc[doc_id].append(page_index)
        for pred in preds:
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
                        "started_with_b": pred["started_with_b"],
                    }
                )
        print(
            example.document_id,
            "production_spans",
            len(production),
            "seqeval_spans",
            len(preds),
            flush=True,
        )

    custom = entity_f1_report(gold_token_rows, pred_token_rows)
    custom.pop("detail", None)
    box_coords = [
        coord
        for sample in pred_box_samples
        for coord in sample["box_returned_by_model"]
    ]
    payload = json.loads(OUT.read_text(encoding="utf-8"))
    payload["val_custom_seqeval"] = custom
    payload["val_span_match_iou_0.5"] = {
        "correct": len(correct),
        "missed": len(missed),
        "false_positives": false_positives,
        "production_spans_requiring_b_tag": production_spans,
        "seqeval_style_spans": seqeval_spans,
        "note": (
            "The saved inference code only starts a field on a B- tag. "
            "This checkpoint's supervised predictions are I- tags, which seqeval "
            "still counts as entities and which are listed here."
        ),
    }
    payload["predicted_word_tags"] = dict(tag_counts)
    payload["correct_fields"] = _spread(correct, 10)
    payload["missed_fields"] = _spread(missed, 10)
    payload["bbox_format"] = {
        "pipeline_converts_to_docile_relative": False,
        "stored_coordinate_space": "0-1000 LayoutLM integers, union of predicted word boxes",
        "docile_coordinate_space": "0-1 [x0, y0, x1, y1] plus a 0-based page index",
        "sample_max_coordinate": max(box_coords) if box_coords else None,
        "samples": pred_box_samples,
        "pages_scored": {doc_id: sorted(pages) for doc_id, pages in pages_by_doc.items()},
    }
    payload["official_evaluator"] = run_official(pages_by_doc, doc_predictions)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}", flush=True)
    print(
        "correct",
        len(correct),
        "missed",
        len(missed),
        "fp",
        false_positives,
        "production",
        production_spans,
        "seqeval_spans",
        seqeval_spans,
        flush=True,
    )


def run_official(pages_by_doc, doc_predictions) -> dict:
    sys.path.insert(0, str(DOCILE_SRC))
    try:
        from docile.dataset import BBox, Dataset, Field
        from docile.dataset.cached_object import CachingConfig
        from docile.evaluation.evaluate import evaluate_dataset
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    docids = list(pages_by_doc)
    predictions = {}
    for doc_id in docids:
        predictions[doc_id] = [
            Field(
                bbox=BBox(*pred["bbox"]),
                page=pred["page"],
                score=pred["score"],
                text=pred["text"],
                fieldtype=pred["fieldtype"],
            )
            for pred in doc_predictions.get(doc_id, [])
        ]
    try:
        dataset = Dataset(
            "invoice-pilot",
            DATA,
            load_annotations=True,
            load_ocr=True,
            cache_images=CachingConfig.OFF,
            docids=docids,
        )
        unscored_pages = {}
        for document in dataset:
            gold_pages = {field.page for field in document.annotation.fields}
            missing = sorted(gold_pages.difference(pages_by_doc[document.docid]))
            if missing:
                unscored_pages[document.docid] = missing
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
        kept = ("AP", "f1", "precision", "recall", "TP", "FP", "FN")
        return {
            "all_kile_fields_on_these_documents": {key: overall[key] for key in kept},
            "nine_trained_fields_micro": {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "TP": tp,
                "FP": fp,
                "FN": fn,
            },
            "per_field": {
                name: {key: row[key] for key in kept} for name, row in per_field.items()
            },
        }

    return {
        "available": True,
        "documents": len(docids),
        "predictions": sum(len(v) for v in predictions.values()),
        "pages_with_gold_but_no_prediction": unscored_pages,
        "iou_1.0": pack(result_strict),
        "iou_0.5": pack(result_debug),
        "iou_1.0_same_text": pack(result_strict, same_text=True)["nine_trained_fields_micro"],
    }


if __name__ == "__main__":
    main()
