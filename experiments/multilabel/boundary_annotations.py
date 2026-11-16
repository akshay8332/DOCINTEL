"""Count nearby labels the party-field boxes tell the model to exclude."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "experiments"))

from docintel.data.docile import _page_fields, _page_words, _split_ids
from multilabel import FIELD_NAMES
from multilabel.data import write_json

PARTY = (
    "vendor_name",
    "customer_billing_name",
    "vendor_address",
    "customer_billing_address",
)
ROLE = {
    "attn",
    "attn:",
    "bill",
    "to",
    "advertiser",
    "agency",
    "accounts",
    "payable",
    "account",
    "station",
    "phone",
    "fax",
    "tel",
    "from",
    "sold",
    "remit",
}
BANDS = (0.02, 0.04, 0.08)


def _overlap_x(left: list[float], right: list[float]) -> bool:
    return min(left[2], right[2]) - max(left[0], right[0]) > 0


def _inside(box: list[float], word_box: list[float]) -> bool:
    cx = (word_box[0] + word_box[2]) / 2
    cy = (word_box[1] + word_box[3]) / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def _empty() -> dict:
    return {
        "pages": 0,
        "party_fields": 0,
        "words_inside_party_fields": 0,
        "nearby_above": {str(band): 0 for band in BANDS},
        "nearby_above_pure_negative": {str(band): 0 for band in BANDS},
        "nearby_above_other_party_field": {str(band): 0 for band in BANDS},
        "role_tokens_inside_party_fields": Counter(),
        "role_tokens_above_0.08": Counter(),
        "top_tokens_above_0.08": Counter(),
    }


def _add_page(stats: dict, words: list[tuple[str, list[float]]], fields: list[tuple[str, list[float]]]) -> None:
    party = [(name, box) for name, box in fields if name in PARTY]
    if not party:
        stats["pages"] += 1
        return
    stats["pages"] += 1
    stats["party_fields"] += len(party)
    membership = []
    for _text, box in words:
        membership.append({name for name, field_box in party if _inside(field_box, box)})
    stats["words_inside_party_fields"] += sum(1 for names in membership if names)
    seen = {band: set() for band in BANDS}
    for name, field_box in party:
        for index, (text, word_box) in enumerate(words):
            if name in membership[index] or not _overlap_x(field_box, word_box):
                continue
            gap = field_box[1] - word_box[3]
            if gap < -0.005:
                continue
            for band in BANDS:
                if gap <= band and index not in seen[band]:
                    seen[band].add(index)
                    key = str(band)
                    stats["nearby_above"][key] += 1
                    if membership[index]:
                        stats["nearby_above_other_party_field"][key] += 1
                    else:
                        stats["nearby_above_pure_negative"][key] += 1
                        if band == 0.08:
                            token = text.lower()
                            stats["top_tokens_above_0.08"][text] += 1
                            if token in ROLE:
                                stats["role_tokens_above_0.08"][text] += 1
    for index, (text, _box) in enumerate(words):
        if membership[index] and text.lower() in ROLE:
            stats["role_tokens_inside_party_fields"][text] += 1


def _scan(root: Path, doc_ids: list[str], page_limit: int | None) -> dict:
    stats = _empty()
    for doc_id in doc_ids:
        if page_limit is not None and stats["pages"] >= page_limit:
            break
        ocr_path = root / "ocr" / f"{doc_id}.json"
        ann_path = root / "annotations" / f"{doc_id}.json"
        if not ocr_path.is_file() or not ann_path.is_file():
            continue
        ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
        annotation = json.loads(ann_path.read_text(encoding="utf-8"))
        for page in ocr.get("pages") or []:
            if page_limit is not None and stats["pages"] >= page_limit:
                break
            page_index = int(page.get("page_idx", 0))
            words = _page_words(page)
            if not words:
                continue
            _add_page(stats, words, _page_fields(annotation, page_index))
    return stats


def _public(stats: dict) -> dict:
    above = stats["top_tokens_above_0.08"]
    return {
        "pages": stats["pages"],
        "party_fields": stats["party_fields"],
        "words_inside_party_fields": stats["words_inside_party_fields"],
        "nearby_above": stats["nearby_above"],
        "nearby_above_pure_negative": stats["nearby_above_pure_negative"],
        "nearby_above_other_party_field": stats["nearby_above_other_party_field"],
        "role_tokens_inside_party_fields": dict(stats["role_tokens_inside_party_fields"]),
        "role_tokens_above_0.08": dict(stats["role_tokens_above_0.08"]),
        "top_tokens_above_0.08": above.most_common(25),
    }


def main() -> int:
    root = ROOT / "data" / "docile"
    doc_ids = _split_ids(root, "train")
    pilot = _public(_scan(root, doc_ids, 200))
    full = _public(_scan(root, doc_ids, None))
    report = {
        "question": "Do party annotations already mark nearby labels as words to exclude?",
        "pilot_200_pages": pilot,
        "full_training_split": full,
        "full_training_started": False,
    }
    destination = ROOT / "artifacts" / "diagnostics" / "party-boundary-annotations.json"
    write_json(destination, report)
    print(json.dumps({"pilot": pilot, "full_pages": full["pages"], "full_above": full["nearby_above"]}, indent=2))
    print(f"Wrote {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
