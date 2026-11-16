"""DocILE invoices → word boxes with BIO labels for LayoutLM."""

from __future__ import annotations

import json
from pathlib import Path

from docintel.schemas.documents import DocumentMeta, Page, WordBox

# Header fields for the first invoice model. Line items stay unlabeled.
INVOICE_FIELDS: list[str] = [
    "document_id",
    "date_issue",
    "date_due",
    "vendor_name",
    "customer_billing_name",
    "vendor_address",
    "customer_billing_address",
    "amount_due",
    "amount_total_gross",
]

INVOICE_BIO_LABELS: list[str] = ["O"] + [
    tag
    for name in INVOICE_FIELDS
    for tag in (f"B-{name}", f"I-{name}")
]


def layout_box(box: list[float]) -> list[int]:
    """Scale a 0–1 box into LayoutLM's 0–1000 coordinate space."""
    vals = [max(0, min(1000, int(round(float(v) * 1000)))) for v in box]
    if vals[2] <= vals[0]:
        if vals[0] >= 1000:
            vals[0] = 999
        vals[2] = min(1000, vals[0] + 1)
    if vals[3] <= vals[1]:
        if vals[1] >= 1000:
            vals[1] = 999
        vals[3] = min(1000, vals[1] + 1)
    return vals


def _pair_box(geometry: list) -> list[float]:
    """OCR stores a box as [[x0, y0], [x1, y1]]."""
    (x0, y0), (x1, y1) = geometry
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def _area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def label_words(
    words: list[tuple[str, list[float]]],
    fields: list[tuple[str, list[float]]],
    *,
    allowed: set[str] | None = None,
) -> list[str]:
    """Assign BIO tags. A word belongs to the smallest field that contains its center."""
    keep = allowed if allowed is not None else set(INVOICE_FIELDS)
    assigned: list[int | None] = [None] * len(words)
    for wi, (_, wbox) in enumerate(words):
        cx = (wbox[0] + wbox[2]) / 2
        cy = (wbox[1] + wbox[3]) / 2
        best: int | None = None
        best_area: float | None = None
        for fi, (ftype, fbox) in enumerate(fields):
            if ftype not in keep:
                continue
            if fbox[0] <= cx <= fbox[2] and fbox[1] <= cy <= fbox[3]:
                area = _area(fbox)
                if best is None or area < best_area:
                    best = fi
                    best_area = area
        assigned[wi] = best

    labels = ["O"] * len(words)
    seen: set[int] = set()
    for wi, fi in enumerate(assigned):
        if fi is None:
            continue
        ftype = fields[fi][0]
        if fi not in seen:
            labels[wi] = f"B-{ftype}"
            seen.add(fi)
        else:
            labels[wi] = f"I-{ftype}"
    return labels


def _split_ids(root: Path, split: str) -> list[str]:
    path = root / f"{split}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing DocILE split file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must be a list of document ids")
    return [str(item) for item in payload]


def render_pdf_page(pdf_path: Path, page_index: int, dest: Path, *, scale: float = 2.0) -> Path:
    """Rasterize one PDF page. Reuse dest when it already exists."""
    import pypdfium2 as pdfium

    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        if page_index < 0 or page_index >= len(pdf):
            raise IndexError(f"{pdf_path.name} has no page {page_index}")
        page = pdf[page_index]
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil().convert("RGB")
        image.save(dest, format="PNG")
        bitmap.close()
        page.close()
    finally:
        pdf.close()
    return dest


def _page_words(page: dict) -> list[tuple[str, list[float]]]:
    words: list[tuple[str, list[float]]] = []
    for block in page.get("blocks") or []:
        for line in block.get("lines") or []:
            for word in line.get("words") or []:
                text = str(word.get("value", "")).strip()
                geometry = word.get("geometry")
                if not text or not geometry:
                    continue
                words.append((text, _pair_box(geometry)))
    return words


def _page_fields(annotation: dict, page_index: int) -> list[tuple[str, list[float]]]:
    fields: list[tuple[str, list[float]]] = []
    for field in annotation.get("field_extractions") or []:
        if int(field.get("page", 0)) != page_index:
            continue
        bbox = field.get("bbox")
        ftype = str(field.get("fieldtype", ""))
        if not bbox or len(bbox) != 4 or not ftype:
            continue
        x0, y0, x1, y1 = [float(v) for v in bbox]
        fields.append((ftype, [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]))
    return fields


def iter_docile_pages(
    root: str | Path,
    split: str = "train",
    *,
    limit: int | None = None,
    image_dir: str | Path | None = None,
):
    """Yield page documents one at a time. ``limit`` None yields the whole split."""
    root = Path(root)
    images = Path(image_dir) if image_dir else root / "page_images"
    count = 0
    for doc_id in _split_ids(root, split):
        if limit is not None and count >= limit:
            break
        ocr_path = root / "ocr" / f"{doc_id}.json"
        ann_path = root / "annotations" / f"{doc_id}.json"
        pdf_path = root / "pdfs" / f"{doc_id}.pdf"
        if not ocr_path.is_file() or not ann_path.is_file() or not pdf_path.is_file():
            continue
        ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
        annotation = json.loads(ann_path.read_text(encoding="utf-8"))
        for page in ocr.get("pages") or []:
            if limit is not None and count >= limit:
                break
            page_index = int(page.get("page_idx", 0))
            raw_words = _page_words(page)
            if not raw_words:
                continue
            try:
                image_path = render_pdf_page(
                    pdf_path,
                    page_index,
                    images / f"{doc_id}-p{page_index}.png",
                )
            except (IndexError, OSError):
                continue
            tags = label_words(raw_words, _page_fields(annotation, page_index))
            word_boxes = [
                WordBox(text=text, box=layout_box(box), label=tag)
                for (text, box), tag in zip(raw_words, tags)
            ]
            count += 1
            if count % 200 == 0:
                print(f"  loaded {count} pages", flush=True)
            yield DocumentMeta(
                document_id=f"{doc_id}-p{page_index}",
                source="docile",
                split=split if split in {"train", "val", "test"} else None,
                pages=[
                    Page(
                        page_number=page_index + 1,
                        width=1000,
                        height=1000,
                        words=word_boxes,
                        image_path=str(image_path),
                    )
                ],
                annotation_path=str(ann_path),
            )


def load_docile_pages(
    root: str | Path,
    split: str = "train",
    *,
    limit: int | None = None,
    image_dir: str | Path | None = None,
) -> list[DocumentMeta]:
    """Load up to ``limit`` page images with OCR words and invoice BIO labels."""
    return list(
        iter_docile_pages(root, split, limit=limit, image_dir=image_dir)
    )
