"""Measure phase 2 without updating the saved checkpoint.

Inference on the training pages is evaluation only. Validation pages are not
used for gradient updates.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "experiments"))

from docintel.data.docile import INVOICE_BIO_LABELS, INVOICE_FIELDS, load_docile_pages
from docintel.data.transforms import document_to_example
from docintel.models.layoutlm import _spans_from_bio, load_layoutlm_for_token_classification
from docintel.models.metrics import align_predictions, entity_f1_report
from phase2.bio_decode import count_bio_by_field, first_subword_agreement, mapping_errors
from phase2.multilabel import annotation_retention, multilabel_matrix
from phase2.ocr_compare import compare_ocr, merge_page_reports
from phase2.spans import spans_from_word_labels

DATA = ROOT / "data" / "docile"
CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "invoice-layoutlm-200"
OUT = ROOT / "artifacts" / "diagnostics" / "phase2-report.json"
PREDICTIONS = ROOT / "artifacts" / "diagnostics" / "phase2-kile-predictions.json"
DOCILE_SRC = Path(r"C:\Users\DELL\Downloads\docile-main\docile-main")
PADDLE_PYTHON = ROOT / ".venv-paddle" / "Scripts" / "python.exe"
TRAIN_FORWARD_PAGES = 50
AMOUNT_TYPES = {
    "amount_due",
    "amount_paid",
    "amount_total_gross",
    "amount_total_net",
    "amount_total_tax",
}
DATE_TYPES = {"date_issue", "date_due"}


def _batched(value):
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
    if tensor.dim() == 1 or (tensor.dim() == 3 and tensor.shape[0] in (1, 3)):
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 2 and tensor.shape[-1] == 4:
        tensor = tensor.unsqueeze(0)
    return tensor


def _page_index(document_id: str) -> int:
    return int(document_id.rsplit("-p", 1)[1])


def _annotations(doc) -> list[dict]:
    page_index = _page_index(doc.document_id)
    payload = json.loads(Path(doc.annotation_path).read_text(encoding="utf-8"))
    fields = []
    for field in payload.get("field_extractions") or []:
        if int(field.get("page", 0)) != page_index:
            continue
        bbox = field.get("bbox")
        name = str(field.get("fieldtype", ""))
        if not bbox or len(bbox) != 4 or not name:
            continue
        box = [float(value) for value in bbox]
        fields.append(
            {
                "fieldtype": name,
                "text": field.get("text"),
                "bbox": [min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])],
            }
        )
    return fields


def _words_rel(doc) -> list[tuple[str, list[float]]]:
    return [
        (word.text, [value / 1000.0 for value in word.box])
        for word in doc.pages[0].words
    ]


def _add_counts(total: dict[str, Counter], page: dict[str, dict[str, int]]) -> None:
    for name, counts in page.items():
        total.setdefault(name, Counter())
        total[name].update(counts)


def _retention_add(total: dict, page: dict) -> None:
    for name, counts in page.items():
        row = total.setdefault(name, {"annotations": 0, "exclusive": 0, "multi": 0})
        for key in row:
            row[key] += counts[key]


def inspect_pages(docs) -> dict:
    bio = {}
    retention = {}
    multi_words = 0
    labeled_words = 0
    pairs: Counter[str] = Counter()
    ocr_pages = []
    ocr_trained_pages = []
    weak_pages = []
    for doc in docs:
        labels = [word.label or "O" for word in doc.pages[0].words]
        _add_counts(bio, count_bio_by_field(labels, INVOICE_FIELDS))
        fields = _annotations(doc)
        words = _words_rel(doc)
        trained = [field for field in fields if field["fieldtype"] in INVOICE_FIELDS]
        trained_pairs = [(field["fieldtype"], field["bbox"]) for field in trained]
        _retention_add(
            retention,
            annotation_retention(words, trained_pairs, allowed=set(INVOICE_FIELDS)),
        )
        matrix = multilabel_matrix(words, trained_pairs, INVOICE_FIELDS)
        for row in matrix:
            active = [name for name, bit in zip(INVOICE_FIELDS, row) if bit]
            if active:
                labeled_words += 1
            if len(active) > 1:
                multi_words += 1
                active.sort()
                for left in range(len(active)):
                    for right in range(left + 1, len(active)):
                        pairs[f"{active[left]}+{active[right]}"] += 1
        ocr_words = [{"text": text, "box": box} for text, box in words]
        page_report = compare_ocr(
            fields, ocr_words, amount_types=AMOUNT_TYPES, date_types=DATE_TYPES
        )
        ocr_pages.append(page_report)
        ocr_trained_pages.append(
            compare_ocr(
                trained, ocr_words, amount_types=AMOUNT_TYPES, date_types=DATE_TYPES
            )
        )
        if page_report["amount_digit_hits"] < page_report["amount_fields"] or (
            page_report["date_digit_hits"] < page_report["date_fields"]
        ):
            weak_pages.append(doc.document_id)
    bio_out = {name: dict(counts) for name, counts in bio.items()}
    return {
        "pages": len(docs),
        "bio_counts": bio_out,
        "words_with_any_trained_field": labeled_words,
        "words_with_multiple_trained_fields": multi_words,
        "overlapping_field_pairs": dict(pairs.most_common(20)),
        "annotation_retention": retention,
        "docile_ocr_all_fields": merge_page_reports(ocr_pages),
        "docile_ocr_trained_fields": merge_page_reports(ocr_trained_pages),
        "pages_with_amount_or_date_digit_miss": weak_pages[:12],
    }


def _token_strings(label_ids, id2label: dict[int, str]) -> list[str]:
    converted = []
    for label_id in label_ids:
        value = int(label_id)
        converted.append("-100" if value == -100 else id2label[value])
    return converted


def b_tag_survival(bundle, docs, id2label: dict[int, str]) -> dict:
    totals = Counter()
    for doc in docs:
        example = document_to_example(doc, label_list=INVOICE_BIO_LABELS)
        encoding = bundle.processor.tokenizer(
            example.words,
            boxes=example.boxes,
            word_labels=example.label_ids,
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        labels = encoding["labels"]
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        totals.update(
            first_subword_agreement(
                example.word_labels,
                list(encoding.word_ids()),
                _token_strings(labels, id2label),
            )
        )
    return dict(totals)


def head_rows(model) -> list[dict]:
    head = model.classifier
    linear = head.out_proj if hasattr(head, "out_proj") else head
    weight = linear.weight.detach().float().cpu()
    bias = linear.bias.detach().float().cpu()
    rows = []
    for index, name in enumerate(INVOICE_BIO_LABELS):
        rows.append(
            {
                "label": name,
                "weight_norm": round(float(weight[index].norm()), 4),
                "bias": round(float(bias[index]), 4),
            }
        )
    return rows


def _new_logit_bucket() -> dict:
    return {
        "n": 0,
        "argmax_B": 0,
        "argmax_same_I": 0,
        "argmax_O": 0,
        "argmax_other": 0,
        "logit_B": 0.0,
        "logit_I": 0.0,
        "logit_O": 0.0,
        "i_logit_beats_b": 0,
    }


def _finish_bucket(bucket: dict) -> dict:
    count = bucket["n"]
    if not count:
        return bucket
    finished = dict(bucket)
    for key in ("logit_B", "logit_I", "logit_O"):
        finished[key] = round(bucket[key] / count, 4)
    return finished


def predict_documents(bundle, docs, id2label: dict[int, str]) -> dict:
    label2id = {name: index for index, name in enumerate(INVOICE_BIO_LABELS)}
    gold_rows = []
    pred_rows = []
    predicted_tags = Counter()
    gold_b = _new_logit_bucket()
    gold_i = _new_logit_bucket()
    per_field_b = {name: _new_logit_bucket() for name in INVOICE_FIELDS}
    before_spans = 0
    after_spans = 0
    recovered_spans = 0
    doc_predictions = defaultdict(list)
    pages_by_doc = defaultdict(list)
    span_gold = []
    span_pred = []

    for doc in docs:
        example = document_to_example(doc, label_list=INVOICE_BIO_LABELS)
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
        image.close()
        data = dict(encoding)
        pixels = data.get("pixel_values")
        if isinstance(pixels, (list, tuple)) and len(pixels) == 1:
            data["pixel_values"] = pixels[0]
        inputs = {
            key: _batched(data[key])
            for key in ("input_ids", "attention_mask", "bbox", "token_type_ids", "pixel_values")
            if key in data
        }
        with torch.inference_mode():
            logits = bundle.model(**inputs).logits[0]
        pred_ids = logits.argmax(dim=-1).tolist()
        confidences = torch.softmax(logits, dim=-1).max(dim=-1).values.tolist()
        token_labels = data["labels"]
        if hasattr(token_labels, "tolist"):
            token_labels = token_labels.tolist()
        if token_labels and isinstance(token_labels[0], list):
            token_labels = token_labels[0]
        truth, guess = align_predictions([pred_ids], [token_labels], id2label)
        gold_rows.extend(truth)
        pred_rows.extend(guess)

        word_ids = list(encoding.word_ids())
        word_labels = ["O"] * len(example.words)
        word_conf = [0.0] * len(example.words)
        seen: set[int] = set()
        for position, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen or word_id >= len(example.words):
                continue
            seen.add(word_id)
            predicted = id2label[int(pred_ids[position])]
            word_labels[word_id] = predicted
            word_conf[word_id] = float(confidences[position])
            predicted_tags[predicted] += 1
            gold = example.word_labels[word_id]
            if gold == "O" or "-" not in gold:
                continue
            tag, kind = gold.split("-", 1)
            row = logits[position]
            b_index = label2id[f"B-{kind}"]
            i_index = label2id[f"I-{kind}"]
            chosen = int(pred_ids[position])
            bucket = gold_b if tag == "B" else gold_i if tag == "I" else None
            if bucket is None:
                continue
            bucket["n"] += 1
            bucket["logit_B"] += float(row[b_index])
            bucket["logit_I"] += float(row[i_index])
            bucket["logit_O"] += float(row[0])
            bucket["i_logit_beats_b"] += int(float(row[i_index]) > float(row[b_index]))
            if chosen == b_index:
                bucket["argmax_B"] += 1
            elif chosen == i_index:
                bucket["argmax_same_I"] += 1
            elif chosen == 0:
                bucket["argmax_O"] += 1
            else:
                bucket["argmax_other"] += 1
            if tag == "B":
                field_bucket = per_field_b[kind]
                field_bucket["n"] += 1
                field_bucket["logit_B"] += float(row[b_index])
                field_bucket["logit_I"] += float(row[i_index])
                field_bucket["logit_O"] += float(row[0])
                field_bucket["i_logit_beats_b"] += int(float(row[i_index]) > float(row[b_index]))
                if chosen == b_index:
                    field_bucket["argmax_B"] += 1
                elif chosen == i_index:
                    field_bucket["argmax_same_I"] += 1
                elif chosen == 0:
                    field_bucket["argmax_O"] += 1
                else:
                    field_bucket["argmax_other"] += 1

        legacy = _spans_from_bio(example.words, word_labels, word_conf, example.boxes)
        rebuilt = spans_from_word_labels(
            example.words,
            word_labels,
            example.boxes,
            word_conf,
            page_index=_page_index(example.document_id),
        )
        before_spans += len(legacy)
        after_spans += len(rebuilt)
        recovered_spans += sum(1 for span in rebuilt if span.recovered)
        doc_id = example.document_id.rsplit("-p", 1)[0]
        pages_by_doc[doc_id].append(_page_index(example.document_id))
        for span in rebuilt:
            doc_predictions[doc_id].append(span.to_docile_dict())
        span_gold.append(
            spans_from_word_labels(
                example.words,
                example.word_labels,
                example.boxes,
                page_index=_page_index(example.document_id),
            )
        )
        span_pred.append(rebuilt)
        print(example.document_id, "legacy", len(legacy), "recovered", len(rebuilt), flush=True)

    custom = entity_f1_report(gold_rows, pred_rows)
    custom.pop("detail", None)
    return {
        "pages": len(docs),
        "seqeval": custom,
        "predicted_word_tags": dict(predicted_tags),
        "gold_B_positions": _finish_bucket(gold_b),
        "gold_I_positions": _finish_bucket(gold_i),
        "gold_B_by_field": {name: _finish_bucket(bucket) for name, bucket in per_field_b.items()},
        "legacy_decoder_spans": before_spans,
        "robust_decoder_spans": after_spans,
        "spans_started_without_b": recovered_spans,
        "doc_predictions": doc_predictions,
        "pages_by_doc": pages_by_doc,
        "span_match": _span_counts(span_gold, span_pred),
    }


def _span_counts(gold_pages, pred_pages) -> dict:
    per_field = {
        name: {"tp": 0, "fp": 0, "fn": 0} for name in INVOICE_FIELDS
    }
    for gold, pred in zip(gold_pages, pred_pages):
        used = set()
        for item in gold:
            match = None
            best = 0.0
            for index, candidate in enumerate(pred):
                if index in used or candidate.label != item.label:
                    continue
                score = _iou(item.box_0_1000, candidate.box_0_1000)
                if score > best:
                    best = score
                    match = index
            if match is not None and best >= 0.5:
                used.add(match)
                per_field[item.label]["tp"] += 1
            else:
                per_field[item.label]["fn"] += 1
        for index, candidate in enumerate(pred):
            if index not in used and candidate.label in per_field:
                per_field[candidate.label]["fp"] += 1
    for row in per_field.values():
        precision = row["tp"] / (row["tp"] + row["fp"]) if row["tp"] + row["fp"] else 0.0
        recall = row["tp"] / (row["tp"] + row["fn"]) if row["tp"] + row["fn"] else 0.0
        row["precision"] = precision
        row["recall"] = recall
        row["f1"] = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return per_field


def _iou(left: list[int], right: list[int]) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area_left = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    area_right = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = area_left + area_right - inter
    return inter / union if union else 0.0


def official_scores(pages_by_doc, doc_predictions) -> dict:
    sys.path.insert(0, str(DOCILE_SRC))
    from docile.dataset import BBox, Dataset, Field
    from docile.dataset.cached_object import CachingConfig
    from docile.evaluation.evaluate import evaluate_dataset

    docids = list(pages_by_doc)
    predictions = {}
    for doc_id in docids:
        predictions[doc_id] = [
            Field(
                bbox=BBox(*item["bbox"]),
                page=item["page"],
                score=item["score"],
                text=item["text"],
                fieldtype=item["fieldtype"],
            )
            for item in doc_predictions.get(doc_id, [])
        ]
    dataset = Dataset(
        "invoice-phase2",
        DATA,
        load_annotations=True,
        load_ocr=True,
        cache_images=CachingConfig.OFF,
        docids=docids,
    )
    strict = evaluate_dataset(dataset, predictions, {}, iou_threshold=1.0)
    relaxed = evaluate_dataset(dataset, predictions, {}, iou_threshold=0.5)

    def pack(result, same_text: bool = False) -> dict:
        overall = result.get_metrics("kile", same_text=same_text)
        per_field = {
            name: result.get_metrics("kile", same_text=same_text, fieldtype=name)
            for name in INVOICE_FIELDS
        }
        kept = ("AP", "f1", "precision", "recall", "TP", "FP", "FN")
        tp = sum(row["TP"] for row in per_field.values())
        fp = sum(row["FP"] for row in per_field.values())
        fn = sum(row["FN"] for row in per_field.values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        samples = _error_samples(result, same_text=same_text)
        return {
            "all_kile_fields": {key: overall[key] for key in kept},
            "nine_trained_fields": {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "TP": tp,
                "FP": fp,
                "FN": fn,
            },
            "per_field": {name: {key: row[key] for key in kept} for name, row in per_field.items()},
            "false_positive_samples": samples["fp"],
            "false_negative_samples": samples["fn"],
        }

    return {
        "documents": len(docids),
        "predictions": sum(len(items) for items in predictions.values()),
        "iou_1.0": pack(strict),
        "iou_0.5": pack(relaxed),
        "iou_1.0_exact_text": pack(strict, same_text=True),
    }


def _error_samples(result, same_text: bool) -> dict:
    false_positives = []
    false_negatives = []
    for doc_id, matching in result.task_to_docid_to_matching["kile"].items():
        filtered = matching.filter(same_text=same_text)
        for field in filtered.false_positives:
            if field.fieldtype not in INVOICE_FIELDS or len(false_positives) >= 10:
                continue
            false_positives.append(
                {
                    "document_id": doc_id,
                    "field": field.fieldtype,
                    "text": field.text,
                    "score": field.score,
                    "page": field.page,
                }
            )
        for field in filtered.false_negatives:
            if field.fieldtype not in INVOICE_FIELDS or len(false_negatives) >= 10:
                continue
            false_negatives.append(
                {
                    "document_id": doc_id,
                    "field": field.fieldtype,
                    "text": field.text,
                    "page": field.page,
                }
            )
    return {"fp": false_positives, "fn": false_negatives}


def compare_paddle(docs) -> dict:
    recognized = run_paddle([doc.document_id for doc in docs])
    if not recognized.get("available"):
        return recognized
    by_id = {item["document_id"]: item["words"] for item in recognized["pages"]}
    paddle_pages = []
    docile_pages = []
    for doc in docs:
        fields = _annotations(doc)
        words = [{"text": text, "box": box} for text, box in _words_rel(doc)]
        paddle_words = by_id.get(doc.document_id, [])
        paddle_pages.append(
            compare_ocr(fields, paddle_words, amount_types=AMOUNT_TYPES, date_types=DATE_TYPES)
        )
        docile_pages.append(
            compare_ocr(fields, words, amount_types=AMOUNT_TYPES, date_types=DATE_TYPES)
        )
    return {
        "available": True,
        "pages": [doc.document_id for doc in docs],
        "paddle": merge_page_reports(paddle_pages),
        "docile_precomputed_on_same_pages": merge_page_reports(docile_pages),
        "paddle_word_counts": {doc.document_id: len(by_id.get(doc.document_id, [])) for doc in docs},
    }


def run_paddle(document_ids: list[str]) -> dict:
    if not PADDLE_PYTHON.is_file():
        return {"available": False, "reason": "Separate PaddleOCR environment is not installed."}
    jobs = []
    for document_id in document_ids:
        doc_id, page_s = document_id.rsplit("-p", 1)
        image = DATA / "page_images" / f"{doc_id}-p{page_s}.png"
        if image.is_file():
            jobs.append({"document_id": document_id, "image": str(image)})
    if not jobs:
        return {"available": False, "reason": "No rendered pages were available for PaddleOCR."}
    work = ROOT / "artifacts" / "diagnostics"
    spec = work / "phase2-paddle-jobs.json"
    output = work / "phase2-paddle-words.json"
    spec.write_text(json.dumps(jobs), encoding="utf-8")
    completed = subprocess.run(
        [str(PADDLE_PYTHON), str(ROOT / "experiments" / "phase2" / "paddle_worker.py"), str(spec), str(output)],
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if completed.returncode != 0 or not output.is_file():
        return {
            "available": False,
            "reason": completed.stderr[-2000:] or completed.stdout[-2000:] or "PaddleOCR failed.",
        }
    recognized = json.loads(output.read_text(encoding="utf-8"))
    # The caller fills annotation comparison. Return the words here.
    return {"available": True, "pages": recognized, "stderr_tail": completed.stderr[-500:]}


def main() -> None:
    print("Loading pages", flush=True)
    train_docs = load_docile_pages(DATA, "train", limit=200)
    val_docs = load_docile_pages(DATA, "val", limit=20)
    print(f"train {len(train_docs)} val {len(val_docs)}", flush=True)
    train_stats = inspect_pages(train_docs)
    id2label = {index: name for index, name in enumerate(INVOICE_BIO_LABELS)}
    config = json.loads((CHECKPOINT / "config.json").read_text(encoding="utf-8"))
    map_errors = mapping_errors(config["id2label"], INVOICE_BIO_LABELS)
    reverse_errors = mapping_errors(
        {value: key for key, value in config["label2id"].items()},
        INVOICE_BIO_LABELS,
    )

    print("Loading checkpoint for inference only", flush=True)
    bundle = load_layoutlm_for_token_classification(
        str(CHECKPOINT),
        label_list=INVOICE_BIO_LABELS,
        prefer_v3=True,
    )
    bundle.model.eval()
    for parameter in bundle.model.parameters():
        parameter.requires_grad_(False)
    print("Checking B tags after tokenization", flush=True)
    survival = b_tag_survival(bundle, train_docs, id2label)
    print("Scoring train sample and validation pages", flush=True)
    train_predictions = predict_documents(bundle, train_docs[:TRAIN_FORWARD_PAGES], id2label)
    val_predictions = predict_documents(bundle, val_docs, id2label)
    stored = {
        doc_id: items for doc_id, items in val_predictions.pop("doc_predictions").items()
    }
    pages_by_doc = val_predictions.pop("pages_by_doc")
    train_predictions.pop("doc_predictions")
    train_predictions.pop("pages_by_doc")
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.write_text(json.dumps(stored, indent=2), encoding="utf-8")
    print("Official evaluator", flush=True)
    official = official_scores(pages_by_doc, stored)
    report = {
        "checkpoint_updated": False,
        "validation_used_for_gradients": False,
        "label_mapping_errors": map_errors,
        "label2id_errors": reverse_errors,
        "classifier_head": head_rows(bundle.model),
        "train_label_audit": train_stats,
        "b_tags_after_tokenization": survival,
        "train_forward_pages": TRAIN_FORWARD_PAGES,
        "train_predictions": train_predictions,
        "validation_predictions": val_predictions,
        "official_validation": official,
        "legacy_decoder_official_submission": {
            "predictions": 0,
            "reason": "The unchanged decoder starts a field only on a B tag, and this checkpoint emits none.",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}", flush=True)
    paddle_ids = train_stats["pages_with_amount_or_date_digit_miss"][:8]
    paddle_docs = [doc for doc in train_docs if doc.document_id in set(paddle_ids)]
    print("PaddleOCR comparison", flush=True)
    try:
        report["paddle_ocr"] = compare_paddle(paddle_docs)
    except Exception as exc:
        report["paddle_ocr"] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PaddleOCR section updated", flush=True)


if __name__ == "__main__":
    main()
