"""Unit tests: FUNSD parse + BIO label mapping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docintel.data.funsd import (
    FUNSD_BIO_LABELS,
    FUNSD_ENTITY_TYPES,
    label_to_id,
    map_entity_label,
    parse_funsd_annotation,
)
from docintel.data.transforms import document_to_example, normalize_box, prepare_examples


FIXTURE_ANN = {
    "form": [
        {
            "id": 0,
            "text": "Date of birth",
            "box": [100, 50, 300, 80],
            "label": "question",
            "words": [
                {"text": "Date", "box": [100, 50, 150, 80]},
                {"text": "of", "box": [155, 50, 180, 80]},
                {"text": "birth", "box": [185, 50, 300, 80]},
            ],
            "linking": [[0, 1]],
        },
        {
            "id": 1,
            "text": "01/01/1990",
            "box": [320, 50, 450, 80],
            "label": "answer",
            "words": [
                {"text": "01/01/1990", "box": [320, 50, 450, 80]},
            ],
            "linking": [[0, 1]],
        },
        {
            "id": 2,
            "text": "PERSONAL INFO",
            "box": [100, 10, 400, 40],
            "label": "header",
            "words": [
                {"text": "PERSONAL", "box": [100, 10, 250, 40]},
                {"text": "INFO", "box": [260, 10, 400, 40]},
            ],
            "linking": [],
        },
        {
            "id": 3,
            "text": "xxx",
            "box": [10, 400, 40, 420],
            "label": "other",
            "words": [
                {"text": "xxx", "box": [10, 400, 40, 420]},
            ],
            "linking": [],
        },
    ]
}


def test_map_entity_label_bio():
    assert map_entity_label("question", 0) == "B-QUESTION"
    assert map_entity_label("question", 1) == "I-QUESTION"
    assert map_entity_label("answer", 0) == "B-ANSWER"
    assert map_entity_label("ANSWER", 2) == "I-ANSWER"
    assert map_entity_label("header", 0) == "B-HEADER"
    assert map_entity_label("other", 0) == "O"
    assert map_entity_label("other", 5) == "O"


def test_map_entity_label_unknown():
    with pytest.raises(ValueError, match="Unknown FUNSD"):
        map_entity_label("invoice_total", 0)


def test_label_to_id_covers_bio_list():
    mapping = label_to_id()
    assert mapping["O"] == 0
    assert set(mapping) == set(FUNSD_BIO_LABELS)
    assert len(mapping) == 7


def test_entity_types_keys():
    assert set(FUNSD_ENTITY_TYPES) == {"other", "header", "question", "answer"}


def test_parse_funsd_annotation(tmp_path: Path):
    ann = tmp_path / "sample.json"
    ann.write_text(json.dumps(FIXTURE_ANN), encoding="utf-8")
    doc = parse_funsd_annotation(ann, document_id="sample", split="train")

    assert doc.document_id == "sample"
    assert doc.source == "funsd"
    assert doc.split == "train"
    assert len(doc.pages) == 1
    page = doc.pages[0]
    labels = [w.label for w in page.words]
    texts = [w.text for w in page.words]

    assert texts == ["Date", "of", "birth", "01/01/1990", "PERSONAL", "INFO", "xxx"]
    assert labels == [
        "B-QUESTION",
        "I-QUESTION",
        "I-QUESTION",
        "B-ANSWER",
        "B-HEADER",
        "I-HEADER",
        "O",
    ]
    assert page.width >= 450
    assert page.height >= 420


def test_normalize_box():
    assert normalize_box([0, 0, 100, 200], 200, 400) == [0, 0, 500, 500]
    assert normalize_box([0, 0, 200, 400], 200, 400) == [0, 0, 1000, 1000]


def test_normalize_box_rejects_bad_size():
    with pytest.raises(ValueError):
        normalize_box([0, 0, 1, 1], 0, 10)


def test_document_to_example_and_prepare(tmp_path: Path):
    ann = tmp_path / "sample.json"
    ann.write_text(json.dumps(FIXTURE_ANN), encoding="utf-8")
    doc = parse_funsd_annotation(ann, document_id="sample", split="train")
    example = document_to_example(doc)
    assert example.document_id == "sample"
    assert len(example.words) == 7
    assert example.word_labels[0] == "B-QUESTION"
    assert example.label_ids[0] == label_to_id()["B-QUESTION"]
    # boxes normalized to 0-1000
    assert all(0 <= c <= 1000 for box in example.boxes for c in box)

    prepared = prepare_examples([doc])
    assert len(prepared) == 1
    assert prepared[0].words == example.words
