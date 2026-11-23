"""Validation inference, DocILE KILE scoring, and the softmax comparison."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from multilabel import FIELD_NAMES
from multilabel.align import word_probabilities
from multilabel.group import INFERENCE_BASELINE, group_fields
from multilabel.model import MultiLabelLayoutLM


def load_multilabel(checkpoint: Path, device: str) -> tuple[MultiLabelLayoutLM, object, dict]:
    from transformers import LayoutLMv3Config, LayoutLMv3Model, LayoutLMv3Processor

    payload = torch.load(checkpoint / "model.pt", map_location=device, weights_only=False)
    # The frozen weights are already in model.pt. Building an empty base model
    # from the public config avoids downloading those weights a second time.
    config_file = checkpoint / "config.json"
    if not config_file.is_file():
        if payload.get("model_name") != "microsoft/layoutlmv3-base":
            config = LayoutLMv3Config.from_pretrained(payload["model_name"])
        else:
            config_file = Path(__file__).resolve().parent / "layoutlmv3_base_config.json"
            config = LayoutLMv3Config.from_json_file(str(config_file))
    else:
        config = LayoutLMv3Config.from_pretrained(checkpoint)
    encoder = LayoutLMv3Model(config)
    model = MultiLabelLayoutLM(encoder, num_labels=len(payload["field_names"]))
    model.encoder.load_state_dict(payload["encoder"])
    model.classifier.load_state_dict(payload["classifier"])
    model.to(device)
    model.eval()
    processor = LayoutLMv3Processor.from_pretrained(checkpoint, apply_ocr=False)
    return model, processor, payload


def predict_pages(model, pages: list[dict], settings: dict, device: str) -> dict:
    threshold = float(settings.get("threshold", 0.5))
    gap = int(settings.get("spatial_gap", 30))
    doc_predictions = defaultdict(list)
    per_type = Counter()
    started = time.perf_counter()
    for page in pages:
        batch = {
            key: page[key].unsqueeze(0).to(device)
            for key in ("input_ids", "attention_mask", "bbox", "pixel_values")
        }
        with torch.inference_mode():
            logits, _loss = model(**batch)
        probabilities = torch.sigmoid(logits[0]).cpu().tolist()
        word_probs = word_probabilities(probabilities, page["word_ids"], len(page["words"]))
        fields = group_fields(
            page["words"],
            page["boxes"],
            word_probs,
            FIELD_NAMES,
            threshold=threshold,
            page_index=page["page_index"],
            spatial_gap=gap,
            grouping=str(settings.get("grouping", INFERENCE_BASELINE)),
        )
        doc_id = page["document_id"].rsplit("-p", 1)[0]
        for field in fields:
            doc_predictions[doc_id].append(field.to_docile_dict())
            per_type[field.label] += 1
    elapsed = time.perf_counter() - started
    return {
        "predictions": doc_predictions,
        "predicted_fields": dict(per_type),
        "seconds": elapsed,
        "seconds_per_page": elapsed / max(len(pages), 1),
        "grouping": str(settings.get("grouping", INFERENCE_BASELINE)),
    }


def official_kile(settings: dict, pages: list[dict], doc_predictions: dict) -> dict:
    sys.path.insert(0, settings["docile_src"])
    from docile.dataset import BBox, Dataset, Field
    from docile.dataset.cached_object import CachingConfig
    from docile.evaluation.evaluate import evaluate_dataset

    pages_by_doc = defaultdict(list)
    for page in pages:
        doc_id, page_s = page["document_id"].rsplit("-p", 1)
        pages_by_doc[doc_id].append(int(page_s))
    docids = list(pages_by_doc)
    predictions = {}
    for doc_id in docids:
        predictions[doc_id] = [
            Field(
                bbox=BBox(*item["bbox"]),
                page=item["page"],
                score=float(item["score"]),
                text=item["text"],
                fieldtype=item["fieldtype"],
            )
            for item in doc_predictions.get(doc_id, [])
        ]
    if sum(len(items) for items in predictions.values()) == 0:
        return {"predictions": 0, "reason": "The model produced no fields at this threshold."}
    dataset = Dataset(
        "multilabel-val",
        settings["data_dir"],
        load_annotations=True,
        load_ocr=True,
        cache_images=CachingConfig.OFF,
        docids=docids,
    )
    strict = evaluate_dataset(dataset, predictions, {}, iou_threshold=1.0)
    relaxed = evaluate_dataset(dataset, predictions, {}, iou_threshold=0.5)
    return {
        "predictions": sum(len(items) for items in predictions.values()),
        "iou_1.0": _pack(strict),
        "iou_0.5": _pack(relaxed),
        "iou_1.0_exact_text": _pack(strict, same_text=True),
    }


def _pack(result, same_text: bool = False) -> dict:
    kept = ("AP", "f1", "precision", "recall", "TP", "FP", "FN")
    overall = result.get_metrics("kile", same_text=same_text)
    per_field = {
        name: result.get_metrics("kile", same_text=same_text, fieldtype=name)
        for name in FIELD_NAMES
    }
    tp = sum(row["TP"] for row in per_field.values())
    fp = sum(row["FP"] for row in per_field.values())
    fn = sum(row["FN"] for row in per_field.values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
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
    }


def extraction_scorecard(official: dict) -> dict:
    """AP, F1, and per-field scores from one official KILE result."""

    def block(key: str) -> dict:
        packed = (official or {}).get(key) or {}
        nine = packed.get("nine_trained_fields") or {}
        per_field = {
            name: {
                "precision": row.get("precision"),
                "recall": row.get("recall"),
                "f1": row.get("f1"),
                "AP": row.get("AP"),
            }
            for name, row in (packed.get("per_field") or {}).items()
        }
        return {
            "ap": (packed.get("all_kile_fields") or {}).get("AP"),
            "nine_field_f1": nine.get("f1"),
            "nine_field_precision": nine.get("precision"),
            "nine_field_recall": nine.get("recall"),
            "per_field": per_field,
        }

    overlap_1 = block("iou_1.0")
    overlap_05 = block("iou_0.5")
    exact = block("iou_1.0_exact_text")
    return {
        "ap_overlap_1.0": overlap_1["ap"],
        "f1_overlap_1.0": overlap_1["nine_field_f1"],
        "f1_overlap_0.5": overlap_05["nine_field_f1"],
        "f1_exact_text": exact["nine_field_f1"],
        "overlap_1.0": overlap_1,
        "overlap_0.5": overlap_05,
        "exact_text": exact,
    }


def softmax_reference(settings: dict) -> dict:
    path = Path(settings["softmax_report"])
    if not path.is_file():
        return {"available": False}
    report = json.loads(path.read_text(encoding="utf-8"))
    official = report["official_validation"]
    return {
        "available": True,
        "token_f1_micro": report["validation_predictions"]["seqeval"]["f1_micro"],
        "train_loss": 0.6102007102966308,
        "iou_1.0": official["iou_1.0"],
        "iou_0.5": official["iou_0.5"],
        "iou_1.0_exact_text": official["iou_1.0_exact_text"],
        "legacy_decoder_predictions": 0,
        "robust_decoder_predictions": official["predictions"],
    }


def comparison_report(settings: dict, multilabel: dict, history: dict) -> dict:
    softmax = softmax_reference(settings)
    focus = [
        "amount_due",
        "amount_total_gross",
        "vendor_name",
        "vendor_address",
        "customer_billing_name",
        "customer_billing_address",
    ]

    def focus_rows(block):
        if not block:
            return {}
        return {name: block["per_field"][name] for name in focus}

    return {
        "same_documents": {
            "train_pages": settings["train_limit"],
            "val_pages": settings["eval_limit"],
            "validation_used_for_gradients": False,
            "threshold": settings["threshold"],
            "threshold_tuned_on_test": False,
        },
        "softmax": softmax,
        "sigmoid": {
            "best_val_loss": history.get("best_val_loss"),
            "epochs_completed": history.get("epochs_completed"),
            "stopped_early": history.get("stopped_early"),
            "final_train_loss": history.get("final_train_loss"),
            "predicted_fields": multilabel.get("predicted_fields"),
            "seconds_per_page": multilabel.get("seconds_per_page"),
            "official": multilabel.get("official"),
        },
        "focus_iou_0.5": {
            "softmax": focus_rows(softmax.get("iou_0.5")) if softmax.get("available") else {},
            "sigmoid": focus_rows((multilabel.get("official") or {}).get("iou_0.5")),
        },
    }
