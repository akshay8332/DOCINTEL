"""Compare a page reader's address text with DocILE on the same gold boxes.

The reader is supplied as word boxes. This script does not train, does not
load a checkpoint, and does not replace DocILE OCR.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("vendor_address", "customer_billing_address")


def _norm_seq(text: str) -> list[str]:
    from multilabel.address_text_audit import norm_token, tokens

    return [token for token in (norm_token(part) for part in tokens(text or "")) if token]


def _same(left: str, right: str, normalized: bool) -> bool:
    if normalized:
        return _norm_seq(left) == _norm_seq(right)
    return (left or "") == (right or "")


def text_inside(words: list[dict], gold_box: list[float]) -> str:
    from docintel.data.docile import layout_box
    from multilabel.address_text_audit import center_inside
    from multilabel.group import assemble_text

    chosen: list[str] = []
    boxes: list[list[int]] = []
    for word in words:
        laid = layout_box([float(value) for value in word["box"]])
        if center_inside(gold_box, laid):
            chosen.append(str(word["text"]))
            boxes.append(laid)
    return assemble_text(chosen, boxes)


def _gold_fields(annotation_path: Path, page_index: int) -> list[dict]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    fields = []
    for field in payload.get("field_extractions") or []:
        if field.get("fieldtype") not in FIELDS or int(field.get("page", 0)) != page_index:
            continue
        bbox = field.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        box = [float(value) for value in bbox]
        fields.append(
            {
                "field": field["fieldtype"],
                "text": field.get("text") or "",
                "bbox": [min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])],
            }
        )
    return fields


def prepare_jobs(data_dir: Path, limit: int) -> dict:
    from docintel.data.docile import load_docile_pages

    documents = load_docile_pages(data_dir, "val", limit=limit)
    pages = []
    for document in documents:
        doc_id, page_s = document.document_id.rsplit("-p", 1)
        page_index = int(page_s)
        image = document.pages[0].image_path
        pages.append(
            {
                "document_id": document.document_id,
                "doc_id": doc_id,
                "page": page_index,
                "image": image,
                "fields": _gold_fields(Path(document.annotation_path), page_index),
            }
        )
    return {
        "validation_pages": len(pages),
        "documents": len({page["doc_id"] for page in pages}),
        "ocr_replaced": False,
        "checkpoint_updated": False,
        "annotations_changed": False,
        "pages": pages,
    }


def _docile_words(data_dir: Path, doc_id: str, page_index: int, cache: dict) -> list[dict]:
    key = (doc_id, page_index)
    if key in cache:
        return cache[key]
    from docintel.data.docile import _page_words

    payload = json.loads((data_dir / "ocr" / f"{doc_id}.json").read_text(encoding="utf-8"))
    page = next(item for item in payload.get("pages") or [] if int(item.get("page_idx", 0)) == page_index)
    words = [{"text": text, "box": box} for text, box in _page_words(page)]
    cache[key] = words
    return words


def _empty_counts() -> dict:
    return {
        "gold": 0,
        "docile_strict": 0,
        "reader_strict": 0,
        "docile_normalized": 0,
        "reader_normalized": 0,
        "reader_only_strict": 0,
        "docile_only_strict": 0,
        "reader_empty": 0,
    }


def score_reader(jobs: dict, reader_pages: dict[str, list[dict]], data_dir: Path) -> dict:
    cache: dict = {}
    counts = {name: _empty_counts() for name in FIELDS}
    gained = {name: [] for name in FIELDS}
    lost = {name: [] for name in FIELDS}
    for page in jobs["pages"]:
        reader_words = reader_pages.get(page["document_id"], [])
        docile_words = _docile_words(data_dir, page["doc_id"], page["page"], cache)
        for field in page["fields"]:
            gold = field["text"]
            docile_text = text_inside(docile_words, field["bbox"])
            reader_text = text_inside(reader_words, field["bbox"])
            row = counts[field["field"]]
            row["gold"] += 1
            docile_strict = _same(docile_text, gold, False)
            reader_strict = _same(reader_text, gold, False)
            row["docile_strict"] += int(docile_strict)
            row["reader_strict"] += int(reader_strict)
            row["docile_normalized"] += int(_same(docile_text, gold, True))
            row["reader_normalized"] += int(_same(reader_text, gold, True))
            row["reader_empty"] += int(not reader_text)
            sample = {
                "document_id": page["doc_id"],
                "page": page["page"],
                "gold": gold,
                "docile": docile_text,
                "reader": reader_text,
            }
            if reader_strict and not docile_strict and len(gained[field["field"]]) < 3:
                gained[field["field"]].append(sample)
            if docile_strict and not reader_strict and len(lost[field["field"]]) < 3:
                lost[field["field"]].append(sample)
            if reader_strict and not docile_strict:
                row["reader_only_strict"] += 1
            if docile_strict and not reader_strict:
                row["docile_only_strict"] += 1
    return {"fields": counts, "gained": gained, "lost": lost}


def render_report(report: dict) -> str:
    lines = [
        f"# Address reader comparison: {report['reader']}",
        "",
        "Same 100 validation pages. The reader saw each official address area. Words whose centers fall inside the official box are joined with the same line breaks as DocILE.",
        "Strict equality is the official exact string. Normalized equality ignores punctuation and line breaks.",
        "DocILE was not replaced, and the checkpoint was not changed.",
        "",
        f"Reader: {report['reader']}. Version: {report.get('ocr_version')}.",
        f"Detection model: {report.get('text_detection_model_name')}. Recognition model: {report.get('text_recognition_model_name')}.",
        f"Pages with an address: {report['pages_read']} of {report['validation_pages']}.",
        "",
        "| Field | Gold | DocILE strict | Reader strict | DocILE normalized | Reader normalized | Reader fixes | Reader breaks |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in FIELDS:
        row = report["fields"][name]
        lines.append(
            "| {name} | {gold} | {docile} | {reader} | {docile_norm} | {reader_norm} | {fixes} | {breaks} |".format(
                name=name,
                gold=row["gold"],
                docile=row["docile_strict"],
                reader=row["reader_strict"],
                docile_norm=row["docile_normalized"],
                reader_norm=row["reader_normalized"],
                fixes=row["reader_only_strict"],
                breaks=row["docile_only_strict"],
            )
        )
    lines.extend(["", "## Examples", ""])
    for name in FIELDS:
        lines.append(f"### {name}")
        lines.append("")
        for label, key in (("Reader matches, DocILE does not", "gained"), ("DocILE matches, reader does not", "lost")):
            samples = report[key][name]
            if not samples:
                continue
            lines.append(f"#### {label}")
            lines.append("")
            for sample in samples:
                lines.append(f"- `{sample['document_id']}` page {sample['page']}")
                lines.append(f"  - gold: {sample['gold']!r}")
                lines.append(f"  - docile: {sample['docile']!r}")
                lines.append(f"  - reader: {sample['reader']!r}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_crops(jobs: dict, dest: Path, *, pad: float = 0.01) -> dict:
    from PIL import Image

    dest.mkdir(parents=True, exist_ok=True)
    crops = []
    for page in jobs["pages"]:
        if not page["fields"]:
            continue
        image = Image.open(page["image"])
        width, height = image.size
        for index, field in enumerate(page["fields"]):
            x0, y0, x1, y1 = field["bbox"]
            x0 = max(0.0, x0 - pad)
            y0 = max(0.0, y0 - pad)
            x1 = min(1.0, x1 + pad)
            y1 = min(1.0, y1 + pad)
            left, top = int(x0 * width), int(y0 * height)
            right, bottom = max(left + 1, int(x1 * width)), max(top + 1, int(y1 * height))
            right, bottom = min(width, right), min(height, bottom)
            crop = image.crop((left, top, right, bottom))
            name = f"{page['document_id']}-{index}.png"
            crop.save(dest / name)
            crops.append(
                {
                    "document_id": page["document_id"],
                    "crop_id": f"{page['document_id']}#{index}",
                    "field": field["field"],
                    "image": str(dest / name),
                    "origin": [left / width, top / height, right / width, bottom / height],
                }
            )
        image.close()
    return {"pages": crops}


def _load_reader(path: Path) -> tuple[dict, dict[str, list[dict]]]:
    meta = {}
    pages: dict[str, list[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") == "meta":
            meta = row
            continue
        pages.setdefault(row["document_id"], []).extend(row.get("words") or [])
    return meta, pages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "crops", "score"))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "docile")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--jobs",
        type=Path,
        default=ROOT / "artifacts" / "diagnostics" / "address-reader-jobs-100.json",
    )
    parser.add_argument("--words", type=Path)
    parser.add_argument("--reader", default="reader")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from multilabel.checkpoint import assert_diagnostic_destination

    if args.command == "prepare":
        assert_diagnostic_destination(args.jobs)
        report = prepare_jobs(args.data_dir, args.limit)
        args.jobs.parent.mkdir(parents=True, exist_ok=True)
        args.jobs.write_text(json.dumps(report), encoding="utf-8")
        addresses = sum(len(page["fields"]) for page in report["pages"])
        print(f"pages {report['validation_pages']} documents {report['documents']} addresses {addresses}")
        print(f"Wrote {args.jobs}")
        return 0
    if args.command == "crops":
        jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
        dest = args.output or (args.jobs.parent / "address-reader-crops")
        crop_jobs = dest.with_suffix(".json") if dest.suffix else dest.parent / "address-reader-crops.json"
        if dest.suffix:
            crop_jobs = dest
            dest = dest.with_name(dest.stem + "-images")
        assert_diagnostic_destination(crop_jobs)
        written = write_crops(jobs, dest)
        crop_jobs.write_text(json.dumps(written), encoding="utf-8")
        print(f"crops {len(written['pages'])}")
        print(f"Wrote {crop_jobs}")
        return 0
    if args.words is None or args.output is None:
        raise SystemExit("score requires --words and --output")
    assert_diagnostic_destination(args.output)
    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
    meta, reader_pages = _load_reader(args.words)
    scored = score_reader(jobs, reader_pages, args.data_dir)
    report = {
        "reader": args.reader,
        "ocr_version": meta.get("ocr_version"),
        "text_detection_model_name": meta.get("text_detection_model_name"),
        "text_recognition_model_name": meta.get("text_recognition_model_name"),
        "validation_pages": jobs["validation_pages"],
        "documents": jobs["documents"],
        "pages_read": len(reader_pages),
        "ocr_replaced": False,
        "checkpoint_updated": False,
        "annotations_changed": False,
        **scored,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = args.output.with_suffix(".md")
    summary.write_text(render_report(report), encoding="utf-8")
    print(summary.read_text(encoding="utf-8"))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
