"""FUNSD download, parse, and BIO label mapping."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlretrieve

from docintel.schemas.documents import DocumentMeta, Page, WordBox

# Official FUNSD mirror used widely by HF / LayoutLM tutorials
FUNSD_ZIP_URL = (
    "https://guillaumejaume.github.io/FUNSD/dataset.zip"
)

FUNSD_ENTITY_TYPES: dict[str, str] = {
    "other": "O",
    "header": "HEADER",
    "question": "QUESTION",
    "answer": "ANSWER",
}

FUNSD_BIO_LABELS: list[str] = [
    "O",
    "B-HEADER",
    "I-HEADER",
    "B-QUESTION",
    "I-QUESTION",
    "B-ANSWER",
    "I-ANSWER",
]


def label_to_id(label_list: list[str] | None = None) -> dict[str, int]:
    labels = label_list or FUNSD_BIO_LABELS
    return {label: idx for idx, label in enumerate(labels)}


def id_to_label(label_list: list[str] | None = None) -> dict[int, str]:
    labels = label_list or FUNSD_BIO_LABELS
    return {idx: label for idx, label in enumerate(labels)}


def map_entity_label(entity_label: str, word_index: int) -> str:
    """Map FUNSD entity type + word position → BIO tag."""
    key = entity_label.strip().lower()
    if key not in FUNSD_ENTITY_TYPES:
        raise ValueError(f"Unknown FUNSD entity label: {entity_label!r}")
    mapped = FUNSD_ENTITY_TYPES[key]
    if mapped == "O":
        return "O"
    prefix = "B" if word_index == 0 else "I"
    return f"{prefix}-{mapped}"


def _annotation_dir(root: Path, split: str) -> Path:
    """Resolve FUNSD annotation folder for train/test."""
    # Official zip layout: dataset/training_data|testing_data/annotations
    candidates = [
        root / "dataset" / f"{split}ing_data" / "annotations",
        root / f"{split}ing_data" / "annotations",
        root / split / "annotations",
    ]
    # train → training_data; test → testing_data
    if split == "train":
        candidates = [
            root / "dataset" / "training_data" / "annotations",
            root / "training_data" / "annotations",
            root / "train" / "annotations",
        ]
    elif split == "test":
        candidates = [
            root / "dataset" / "testing_data" / "annotations",
            root / "testing_data" / "annotations",
            root / "test" / "annotations",
        ]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"No FUNSD annotations for split={split!r} under {root}. "
        "Run scripts/download_funsd.py first."
    )


def _image_dir_for_annotations(ann_dir: Path) -> Path:
    sibling = ann_dir.parent / "images"
    if sibling.is_dir():
        return sibling
    raise FileNotFoundError(f"Missing images directory next to {ann_dir}")


def parse_funsd_annotation(
    path: str | Path,
    *,
    image_path: str | Path | None = None,
    document_id: str | None = None,
    split: str | None = None,
    page_width: int = 1000,
    page_height: int = 1000,
) -> DocumentMeta:
    """Parse one FUNSD annotation JSON into DocumentMeta with BIO word labels."""
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        payload: dict[str, Any] = json.load(fh)

    form: Iterable[dict[str, Any]] = payload.get("form", [])
    words: list[WordBox] = []
    for entity in form:
        entity_label = str(entity.get("label", "other"))
        entity_words = entity.get("words") or []
        if not entity_words and entity.get("text"):
            entity_words = [{"text": entity["text"], "box": entity.get("box", [0, 0, 0, 0])}]
        for idx, word in enumerate(entity_words):
            text = str(word.get("text", "")).strip()
            if not text:
                continue
            box = [int(v) for v in word["box"]]
            bio = map_entity_label(entity_label, idx)
            words.append(WordBox(text=text, box=box, label=bio))

    # Infer page size from max box extents when possible
    if words:
        page_width = max(page_width, max(w.box[2] for w in words))
        page_height = max(page_height, max(w.box[3] for w in words))

    doc_id = document_id or path.stem
    page = Page(
        page_number=1,
        width=page_width,
        height=page_height,
        words=words,
        image_path=str(image_path) if image_path else None,
    )
    return DocumentMeta(
        document_id=doc_id,
        source="funsd",
        split=split,  # type: ignore[arg-type]
        pages=[page],
        annotation_path=str(path),
    )


def load_funsd_split(
    root: str | Path,
    split: str = "train",
) -> list[DocumentMeta]:
    """Load all FUNSD documents for a split (train or test)."""
    root = Path(root)
    ann_dir = _annotation_dir(root, split)
    img_dir = _image_dir_for_annotations(ann_dir)
    docs: list[DocumentMeta] = []
    for ann_path in sorted(ann_dir.glob("*.json")):
        stem = ann_path.stem
        image = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                image = candidate
                break
        docs.append(
            parse_funsd_annotation(
                ann_path,
                image_path=image,
                document_id=stem,
                split="train" if split == "train" else "test",
            )
        )
    return docs


def download_funsd(
    dest: str | Path,
    *,
    url: str = FUNSD_ZIP_URL,
    force: bool = False,
) -> Path:
    """Download and extract FUNSD into ``dest`` (creates dataset/ layout)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / "dataset" / "training_data" / "annotations"
    if marker.is_dir() and not force:
        return dest

    zip_path = dest / "dataset.zip"
    print(f"Downloading FUNSD from {url} -> {zip_path}")
    urlretrieve(url, zip_path)
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    # Some mirrors nest an extra folder; normalize if needed
    nested = dest / "dataset"
    if not nested.exists():
        # find first dir containing training_data
        for child in dest.iterdir():
            if child.is_dir() and (child / "training_data").exists():
                shutil.move(str(child), str(nested))
                break
    print(f"FUNSD ready at {dest}")
    return dest


def cli_download() -> None:
    parser = argparse.ArgumentParser(description="Download FUNSD dataset")
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("./data/funsd"),
        help="Destination directory",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_funsd(args.dest, force=args.force)


if __name__ == "__main__":
    cli_download()
