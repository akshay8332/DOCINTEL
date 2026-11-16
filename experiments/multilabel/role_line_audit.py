"""Audit role lines inside and outside party-field boxes. Annotations are not changed."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "experiments"))

from docintel.data.docile import _page_fields, _pair_box, _split_ids
from multilabel.data import write_json

PARTY = (
    "vendor_name",
    "customer_billing_name",
    "vendor_address",
    "customer_billing_address",
)
PHRASES = (
    ("attn", ("attn",)),
    ("accounts_payable", ("accounts", "payable")),
    ("bill_to", ("bill", "to")),
    ("remit", ("remit",)),
    ("remit_to", ("remit", "to")),
    ("advertiser", ("advertiser",)),
    ("agency", ("agency",)),
)


def _token(text: str) -> str:
    return "".join(char.lower() for char in text if char.isalnum())


def _lines(page: dict) -> list[list[tuple[str, list[float]]]]:
    lines = []
    for block in page.get("blocks") or []:
        for line in block.get("lines") or []:
            words = []
            for word in line.get("words") or []:
                text = str(word.get("value", "")).strip()
                geometry = word.get("geometry")
                if not text or not geometry:
                    continue
                words.append((text, _pair_box(geometry)))
            if words:
                lines.append(words)
    return lines


def _union(boxes: list[list[float]]) -> list[float]:
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _inside(box: list[float], word_box: list[float]) -> bool:
    cx = (word_box[0] + word_box[2]) / 2
    cy = (word_box[1] + word_box[3]) / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def _overlap_x(left: list[float], right: list[float]) -> bool:
    return min(left[2], right[2]) - max(left[0], right[0]) > 0


def _find(line: list[tuple[str, list[float]]], parts: tuple[str, ...]) -> list[int]:
    tokens = [_token(text) for text, _box in line]
    starts = []
    width = len(parts)
    for index in range(len(tokens) - width + 1):
        if tuple(tokens[index : index + width]) == parts:
            starts.append(index)
    return starts


def _relation(phrase_box: list[float], following: list[list[float]], fields: list[tuple[str, list[float]]]) -> dict:
    containing = []
    following_fields = []
    for name, box in fields:
        if _inside(box, phrase_box):
            containing.append(name)
        if following and any(_inside(box, word_box) for word_box in following):
            following_fields.append(name)
    nearest = None
    for name, box in fields:
        if not _overlap_x(phrase_box, box):
            continue
        gap = box[1] - phrase_box[3]
        if gap < -0.005:
            continue
        if nearest is None or gap < nearest[0]:
            nearest = (gap, name)
    return {
        "inside": containing,
        "following_inside": following_fields,
        "nearest_above": None if nearest is None or nearest[0] > 0.08 else {"gap": round(nearest[0], 4), "field": nearest[1]},
    }


def _empty() -> dict:
    return {
        "pages": 0,
        "phrases": {
            name: {
                "occurrences": 0,
                "alone_on_line": Counter(),
                "followed_by_words": Counter(),
                "examples": [],
            }
            for name, _parts in PHRASES
        },
    }


def _place(relation: dict) -> str:
    inside = relation["inside"]
    if inside:
        return "inside:" + "+".join(inside)
    if relation["following_inside"]:
        return "phrase_outside_line_inside:" + "+".join(relation["following_inside"])
    if relation["nearest_above"]:
        return f"above:{relation['nearest_above']['field']}"
    return "no_nearby_party_field"


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
            lines = _lines(page)
            if not lines:
                continue
            stats["pages"] += 1
            fields = [(name, box) for name, box in _page_fields(annotation, page_index) if name in PARTY]
            for line in lines:
                for phrase, parts in PHRASES:
                    for start in _find(line, parts):
                        if phrase == "remit":
                            following_tokens = [_token(text) for text, _box in line[start + 1 :]]
                            if following_tokens and following_tokens[0] == "to":
                                continue
                        chosen = line[start : start + len(parts)]
                        following = [box for _text, box in line[start + len(parts) :]]
                        relation = _relation(_union([box for _text, box in chosen]), following, fields)
                        bucket = stats["phrases"][phrase]
                        bucket["occurrences"] += 1
                        place = _place(relation)
                        key = "alone_on_line" if not following else "followed_by_words"
                        bucket[key][place] += 1
                        if len(bucket["examples"]) < 8 and (
                            place.startswith("phrase_outside") or place.startswith("inside:")
                        ):
                            bucket["examples"].append(
                                {
                                    "document_id": f"{doc_id}-p{page_index}",
                                    "line": " ".join(text for text, _box in line),
                                    "place": place,
                                    "followed_by_words": bool(following),
                                }
                            )
    return stats


def _public(stats: dict) -> dict:
    phrases = {}
    for name, bucket in stats["phrases"].items():
        phrases[name] = {
            "occurrences": bucket["occurrences"],
            "alone_on_line": dict(bucket["alone_on_line"]),
            "followed_by_words": dict(bucket["followed_by_words"]),
            "examples": bucket["examples"],
        }
    return {"pages": stats["pages"], "phrases": phrases}


def main() -> int:
    root = ROOT / "data" / "docile"
    doc_ids = _split_ids(root, "train")
    report = {
        "annotations_modified": False,
        "full_training_started": False,
        "pilot_200_pages": _public(_scan(root, doc_ids, 200)),
        "full_training_split": _public(_scan(root, doc_ids, None)),
    }
    destination = ROOT / "artifacts" / "diagnostics" / "role-line-audit.json"
    write_json(destination, report)
    print(f"Wrote {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
