"""Rebuild name and address text from the frozen checkpoint. Do not save weights."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "experiments"))

from multilabel.data import encode_document, load_settings, prepare_documents, write_json
from multilabel.evaluate import load_multilabel, official_kile, predict_pages
from multilabel.group import MULTILINE_FIELDS

REVIEW_FIELDS = (
    "vendor_name",
    "customer_billing_name",
    "vendor_address",
    "customer_billing_address",
)


def _norm(text: str) -> str:
    return " ".join(text.split())


def _iou(left: list[float], right: list[float]) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - inter
    return inter / union if union else 0.0


def review_predictions(docs, doc_predictions: dict) -> dict:
    """Classify name and address predictions against the annotation boxes."""
    gold = defaultdict(list)
    for doc in docs:
        page = int(doc.document_id.rsplit("-p", 1)[1])
        doc_id = doc.document_id.rsplit("-p", 1)[0]
        payload = json.loads(Path(doc.annotation_path).read_text(encoding="utf-8"))
        for field in payload.get("field_extractions") or []:
            if int(field.get("page", 0)) != page or field.get("fieldtype") not in REVIEW_FIELDS:
                continue
            gold[doc_id].append(
                {
                    "fieldtype": field["fieldtype"],
                    "page": page,
                    "text": field.get("text") or "",
                    "bbox": [float(value) for value in field["bbox"]],
                }
            )

    counts = {name: {"exact": 0, "location_only": 0, "extra": 0} for name in REVIEW_FIELDS}
    examples = {"location_only": [], "extra": []}
    for doc_id, predictions in doc_predictions.items():
        for prediction in predictions:
            name = prediction["fieldtype"]
            if name not in REVIEW_FIELDS:
                continue
            candidates = [
                field
                for field in gold.get(doc_id, [])
                if field["fieldtype"] == name and field["page"] == prediction["page"]
            ]
            overlap = 0.0
            field = None
            for candidate in candidates:
                score = _iou(prediction["bbox"], candidate["bbox"])
                if score > overlap:
                    overlap = score
                    field = candidate
            text = prediction["text"]
            if overlap >= 0.5 and field is not None and text == field["text"]:
                counts[name]["exact"] += 1
                kind = "exact"
            elif overlap >= 0.5 and field is not None:
                counts[name]["location_only"] += 1
                kind = "location_only"
            else:
                counts[name]["extra"] += 1
                kind = "extra"
            if kind != "exact" and len(examples[kind]) < 8:
                examples[kind].append(
                    {
                        "document_id": doc_id,
                        "field": name,
                        "page": prediction["page"],
                        "score": prediction["score"],
                        "overlap": round(overlap, 3),
                        "prediction": text,
                        "gold": None if field is None else field["text"],
                        "same_words": bool(field) and _norm(text) == _norm(field["text"]),
                    }
                )
    return {"counts": counts, "examples": examples, "multiline_fields": sorted(MULTILINE_FIELDS)}


def _slice_report(settings, pages, predictions, start: int) -> dict:
    chosen = pages[start:]
    seen = {page["document_id"].rsplit("-p", 1)[0] for page in pages[:start]}
    held_out = [page for page in chosen if page["document_id"].rsplit("-p", 1)[0] not in seen]
    kept = {page["document_id"].rsplit("-p", 1)[0] for page in held_out}
    subset = {doc_id: predictions[doc_id] for doc_id in kept if doc_id in predictions}
    return {
        "pages": len(held_out),
        "documents": len(kept),
        "official": official_kile(settings, held_out, subset) if held_out else {},
    }


def main() -> int:
    settings = load_settings(ROOT / "configs" / "multilabel-200.yaml")
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else int(settings["eval_limit"])
    checkpoint = Path(settings["weights_checkpoint"])
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Reading frozen checkpoint {checkpoint}", flush=True)
    print(f"Grouping {settings.get('grouping')} pages {limit}", flush=True)
    model, processor, _payload = load_multilabel(checkpoint, device)
    docs = prepare_documents(Path(settings["data_dir"]), "val", limit)
    pages = [
        encode_document(doc, processor, max_length=int(settings["max_seq_length"]))
        for doc in docs
    ]
    predicted = predict_pages(model, pages, settings, device)
    print(f"Predictions {sum(predicted['predicted_fields'].values())}", flush=True)
    official = official_kile(settings, pages, predicted["predictions"])
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_updated": False,
        "grouping": predicted["grouping"],
        "pages": len(pages),
        "threshold": settings["threshold"],
        "predicted_fields": predicted["predicted_fields"],
        "seconds_per_page": predicted["seconds_per_page"],
        "official": official,
        "name_address_review": review_predictions(docs, predicted["predictions"]),
    }
    if limit > int(settings["eval_limit"]):
        report["after_original_validation_pages"] = _slice_report(
            settings, pages, predicted["predictions"], int(settings["eval_limit"])
        )
    destination = ROOT / "artifacts" / "diagnostics"
    name = "multilabel-reconstruction.json" if limit == int(settings["eval_limit"]) else f"multilabel-reconstruction-{limit}.json"
    write_json(destination / name, report)
    print(f"Wrote {destination / name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
