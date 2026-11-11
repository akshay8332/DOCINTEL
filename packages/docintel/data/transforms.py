"""OCR / FUNSD words → LayoutLM feature dicts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from docintel.data.funsd import FUNSD_BIO_LABELS, label_to_id
from docintel.schemas.documents import DocumentMeta, WordBox


@dataclass
class LayoutLMExample:
    document_id: str
    words: list[str]
    boxes: list[list[int]]
    word_labels: list[str]
    label_ids: list[int]
    image_path: str | None = None
    width: int = 1000
    height: int = 1000


def normalize_box(box: Sequence[int], width: int, height: int) -> list[int]:
    """Scale pixel box to LayoutLM 0–1000 coordinate space."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    x0, y0, x1, y1 = box
    return [
        max(0, min(1000, int(1000 * x0 / width))),
        max(0, min(1000, int(1000 * y0 / height))),
        max(0, min(1000, int(1000 * x1 / width))),
        max(0, min(1000, int(1000 * y1 / height))),
    ]


def document_to_example(
    doc: DocumentMeta,
    *,
    label_list: list[str] | None = None,
) -> LayoutLMExample:
    labels = label_list or FUNSD_BIO_LABELS
    lid = label_to_id(labels)
    page = doc.pages[0]
    words: list[str] = []
    boxes: list[list[int]] = []
    word_labels: list[str] = []
    label_ids: list[int] = []
    for w in page.words:
        words.append(w.text)
        boxes.append(normalize_box(w.box, page.width, page.height))
        bio = w.label or "O"
        word_labels.append(bio)
        label_ids.append(lid.get(bio, lid["O"]))
    return LayoutLMExample(
        document_id=doc.document_id,
        words=words,
        boxes=boxes,
        word_labels=word_labels,
        label_ids=label_ids,
        image_path=page.image_path,
        width=page.width,
        height=page.height,
    )


def prepare_examples(
    documents: Sequence[DocumentMeta],
    *,
    label_list: list[str] | None = None,
) -> list[LayoutLMExample]:
    return [document_to_example(d, label_list=label_list) for d in documents]


def encode_example_for_layoutlm(
    example: LayoutLMExample,
    tokenizer: Any,
    *,
    max_length: int = 512,
    label_all_tokens: bool = False,
) -> dict[str, Any]:
    """
    Tokenize words and align word-level BIO labels to subword tokens.

    Works with LayoutLMv2/v3 tokenizers that accept ``boxes=``.
    """
    words = example.words
    boxes = example.boxes
    word_labels = example.label_ids

    encoding = tokenizer(
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_offsets_mapping=False,
        return_tensors=None,
    )

    # word_ids() available on BatchEncoding from Fast tokenizers
    word_ids = encoding.word_ids()
    aligned: list[int] = []
    previous_word: int | None = None
    ignore_index = -100
    for word_idx in word_ids:
        if word_idx is None:
            aligned.append(ignore_index)
        elif word_idx != previous_word:
            aligned.append(word_labels[word_idx])
        else:
            aligned.append(word_labels[word_idx] if label_all_tokens else ignore_index)
        previous_word = word_idx

    encoding["labels"] = aligned
    encoding["document_id"] = example.document_id
    return dict(encoding)


def words_from_boxes(words: Sequence[WordBox]) -> tuple[list[str], list[list[int]]]:
    return [w.text for w in words], [list(w.box) for w in words]
