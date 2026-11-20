"""Count party and name patterns on the official DocILE training split.

Validation and test annotations are not read. No labels are created.
The frozen checkpoint and the trainer are not changed.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))

from docintel.data.docile import _page_words

DATA = ROOT / "data" / "docile"
OUTPUT = ROOT / "artifacts" / "diagnostics" / "party-training-coverage.json"
PARTY = (
    "vendor_name",
    "vendor_address",
    "customer_billing_name",
    "customer_billing_address",
)
COMPANY = {
    "inc", "llc", "ltd", "corp", "corporation", "co", "company", "llp", "plc",
    "group", "bank", "university", "assoc", "association", "limited",
}
ATTENTION = ("attn", "attention", "c/o", "clo ")
HEAD = {"bill", "ship", "sold", "remit", "prepared", "from", "advertiser", "agency"}
_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def _fields(annotation: dict) -> dict[int, list[dict]]:
    pages: dict[int, list[dict]] = {}
    for field in annotation.get("field_extractions") or []:
        name = str(field.get("fieldtype", ""))
        bbox = field.get("bbox")
        if name not in PARTY or not bbox or len(bbox) != 4:
            continue
        box = [float(value) for value in bbox]
        pages.setdefault(int(field.get("page", 0)), []).append(
            {
                "field": name,
                "text": field.get("text") or "",
                "bbox": [min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])],
            }
        )
    return pages


def _overlap(left: list[float], right: list[float]) -> float:
    width = min(left[2], right[2]) - max(left[0], right[0])
    smaller = min(left[2] - left[0], right[2] - right[0])
    if smaller <= 0:
        return 0.0
    return max(0.0, width) / smaller


def _above(name: list[float], address: list[float]) -> bool:
    gap = address[1] - name[3]
    return -0.02 <= gap <= 0.05 and _overlap(name, address) >= 0.3


def _company(text: str) -> bool:
    tokens = [_norm(token) for token in _TOKEN.findall(text)]
    return any(token in COMPANY for token in tokens)


def _person(text: str) -> bool:
    tokens = [_norm(token) for token in _TOKEN.findall(text)]
    tokens = [token for token in tokens if token]
    return 2 <= len(tokens) <= 4 and not _company(text)


def _contains(box: list[float], word_box: list[float]) -> bool:
    cx = (word_box[0] + word_box[2]) / 2
    cy = (word_box[1] + word_box[3]) / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def _heading(words: list[tuple[str, list[float]]], address: list[float]) -> bool:
    chosen = []
    for text, box in words:
        if box[3] > address[1] + 0.008:
            continue
        if address[1] - box[3] > 0.04:
            continue
        if _overlap(box, address) < 0.2 and not (address[0] <= (box[0] + box[2]) / 2 <= address[2]):
            continue
        chosen.append(_norm(text))
    if any(token in HEAD for token in chosen):
        return True
    return any(token in {"to", "for"} for token in chosen) and any(
        token in {"bill", "ship", "sold", "remit", "prepared"} for token in chosen
    )


def main() -> int:
    train_ids = json.loads((DATA / "train.json").read_text(encoding="utf-8"))
    val_ids = set(json.loads((DATA / "val.json").read_text(encoding="utf-8")))
    test_ids = set(json.loads((DATA / "test.json").read_text(encoding="utf-8")))
    overlap = {
        "train_and_val": len(set(train_ids) & val_ids),
        "train_and_test": len(set(train_ids) & test_ids),
        "val_and_test": len(val_ids & test_ids),
    }
    tight = {"heading": set(), "two_column": set(), "attention": set(), "name_above": set(), "both_parties": set()}
    counts = {
        "documents": 0,
        "pages": 0,
        "field_annotations": {name: 0 for name in PARTY},
        "pages_with_field": {name: 0 for name in PARTY},
        "pages_with_both_parties": 0,
        "pages_with_all_four": 0,
        "pages_vendor_name_and_address": 0,
        "pages_customer_name_and_address": 0,
        "name_above_address": {"vendor": 0, "customer": 0},
        "company_names": {"vendor_name": 0, "customer_billing_name": 0},
        "person_like_names": {"vendor_name": 0, "customer_billing_name": 0},
        "attention_in_address_text": {"vendor_address": 0, "customer_billing_address": 0},
        "two_column_address_pages": 0,
        "heading_above_address_pages": 0,
        "words_in_field": {name: 0 for name in PARTY},
        "words_in_name_and_address": {"vendor": 0, "customer": 0},
        "words_in_both_parties": 0,
    }
    manifest = []
    for index, doc_id in enumerate(train_ids, start=1):
        ann_path = DATA / "annotations" / f"{doc_id}.json"
        ocr_path = DATA / "ocr" / f"{doc_id}.json"
        if not ann_path.is_file():
            continue
        annotation = json.loads(ann_path.read_text(encoding="utf-8"))
        pages = _fields(annotation)
        if not pages:
            continue
        counts["documents"] += 1
        ocr_pages = None
        repeat = False
        for page_index, fields in pages.items():
            counts["pages"] += 1
            present = {field["field"] for field in fields}
            for name in present:
                counts["pages_with_field"][name] += 1
            for field in fields:
                counts["field_annotations"][field["field"]] += 1
                if field["field"].endswith("_name"):
                    if _company(field["text"]):
                        counts["company_names"][field["field"]] += 1
                    elif _person(field["text"]):
                        counts["person_like_names"][field["field"]] += 1
                if field["field"].endswith("_address") and any(piece in field["text"].lower() for piece in ATTENTION):
                    counts["attention_in_address_text"][field["field"]] += 1
                    if field["field"] == "customer_billing_address":
                        tight["attention"].add(doc_id)
            vendor_side = "vendor_name" in present or "vendor_address" in present
            customer_side = "customer_billing_name" in present or "customer_billing_address" in present
            if vendor_side and customer_side:
                counts["pages_with_both_parties"] += 1
                tight["both_parties"].add(doc_id)
                repeat = True
            if present >= set(PARTY):
                counts["pages_with_all_four"] += 1
            if "vendor_name" in present and "vendor_address" in present:
                counts["pages_vendor_name_and_address"] += 1
            if "customer_billing_name" in present and "customer_billing_address" in present:
                counts["pages_customer_name_and_address"] += 1
            by_name = {}
            for field in fields:
                by_name.setdefault(field["field"], []).append(field)
            for party, name_key, address_key in (
                ("vendor", "vendor_name", "vendor_address"),
                ("customer", "customer_billing_name", "customer_billing_address"),
            ):
                for name in by_name.get(name_key, []):
                    for address in by_name.get(address_key, []):
                        if _above(name["bbox"], address["bbox"]):
                            counts["name_above_address"][party] += 1
                            tight["name_above"].add(doc_id)
                            repeat = True
                            break
            vendor_boxes = [field["bbox"] for field in by_name.get("vendor_address", [])]
            customer_boxes = [field["bbox"] for field in by_name.get("customer_billing_address", [])]
            columns = False
            for left in vendor_boxes:
                for right in customer_boxes:
                    centers = abs((left[0] + left[2]) / 2 - (right[0] + right[2]) / 2)
                    if _overlap(left, right) < 0.25 and centers > 0.2:
                        columns = True
            if columns:
                counts["two_column_address_pages"] += 1
                tight["two_column"].add(doc_id)
                repeat = True
            if ocr_path.is_file():
                if ocr_pages is None:
                    ocr_pages = {
                        int(page.get("page_idx", 0)): _page_words(page)
                        for page in json.loads(ocr_path.read_text(encoding="utf-8")).get("pages") or []
                    }
                words = ocr_pages.get(page_index, [])
                if any(_heading(words, field["bbox"]) for field in fields if field["field"].endswith("_address")):
                    counts["heading_above_address_pages"] += 1
                    tight["heading"].add(doc_id)
                    repeat = True
                membership = []
                for _text, box in words:
                    hits = {field["field"] for field in fields if _contains(field["bbox"], box)}
                    if hits:
                        membership.append(hits)
                        for name in hits:
                            counts["words_in_field"][name] += 1
                for hits in membership:
                    if "vendor_name" in hits and "vendor_address" in hits:
                        counts["words_in_name_and_address"]["vendor"] += 1
                    if "customer_billing_name" in hits and "customer_billing_address" in hits:
                        counts["words_in_name_and_address"]["customer"] += 1
                    if hits & {"vendor_name", "vendor_address"} and hits & {
                        "customer_billing_name",
                        "customer_billing_address",
                    }:
                        counts["words_in_both_parties"] += 1
        if repeat:
            manifest.append(doc_id)
        if index % 500 == 0:
            print(f"scanned {index}", flush=True)
    report = {
        "split": "train",
        "train_ids": len(train_ids),
        "val_ids_not_read": len(val_ids),
        "test_ids_not_read": len(test_ids),
        "split_overlap": overlap,
        "counts": counts,
        "manifest_documents": len(manifest),
        "manifest_rule": (
            "Train document with a page that has both parties, a name stacked on its address, "
            "two address columns, or a heading word directly above an official address."
        ),
        "targeted_documents": {name: len(ids) for name, ids in tight.items()},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest_path = OUTPUT.with_name("party-training-manifest.json")
    focused = sorted(tight["heading"] | tight["two_column"] | tight["attention"])
    manifest_path.write_text(
        json.dumps(
            {
                "split": "train",
                "broad_document_ids": manifest,
                "focused_document_ids": focused,
                "focused_rule": "Heading above an address, two address columns, or an attention line in a customer address.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    committed = ROOT / "configs" / "party_experiment_focused_ids.json"
    committed.write_text(
        json.dumps(
            {
                "split": "train",
                "document_ids": focused,
                "rule": "Heading above an address, two address columns, or an attention line in a customer address.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"documents": counts["documents"], "pages": counts["pages"], "manifest": len(manifest), "focused": len(focused)}, indent=2))
    print(f"Wrote {OUTPUT}")
    print(f"Wrote {committed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
