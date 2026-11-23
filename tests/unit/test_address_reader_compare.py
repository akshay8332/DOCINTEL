"""Address text inside a gold box. No reader and no checkpoint are loaded."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "packages"))

from multilabel.address_reader_compare import score_reader, text_inside


def test_text_inside_uses_word_centers():
    words = [
        {"text": "Tallahassee,", "box": [0.10, 0.10, 0.30, 0.14]},
        {"text": "FL", "box": [0.32, 0.10, 0.40, 0.14]},
        {"text": "Header", "box": [0.10, 0.50, 0.30, 0.54]},
    ]
    assert text_inside(words, [0.0, 0.0, 0.5, 0.2]) == "Tallahassee, FL"


def test_score_counts_a_reader_fix_without_replacing_docile(tmp_path: Path):
    data = tmp_path / "docile"
    ocr = data / "ocr"
    ocr.mkdir(parents=True)
    ocr.joinpath("doc.json").write_text(
        json_page("Tallahassce,"),
        encoding="utf-8",
    )
    jobs = {
        "pages": [
            {
                "document_id": "doc-p0",
                "doc_id": "doc",
                "page": 0,
                "fields": [
                    {
                        "field": "vendor_address",
                        "text": "Tallahassee, FL",
                        "bbox": [0.0, 0.0, 0.5, 0.2],
                    }
                ],
            }
        ]
    }
    reader = {
        "doc-p0": [
            {"text": "Tallahassee,", "box": [0.10, 0.10, 0.30, 0.14]},
            {"text": "FL", "box": [0.32, 0.10, 0.40, 0.14]},
        ]
    }
    report = score_reader(jobs, reader, data)
    row = report["fields"]["vendor_address"]
    assert row["gold"] == 1
    assert row["docile_strict"] == 0
    assert row["reader_strict"] == 1
    assert row["reader_only_strict"] == 1
    assert report["gained"]["vendor_address"][0]["reader"] == "Tallahassee, FL"


def json_page(city: str) -> str:
    import json

    return json.dumps(
        {
            "pages": [
                {
                    "page_idx": 0,
                    "blocks": [
                        {
                            "lines": [
                                {
                                    "words": [
                                        {"value": city, "geometry": [[0.10, 0.10], [0.30, 0.14]]},
                                        {"value": "FL", "geometry": [[0.32, 0.10], [0.40, 0.14]]},
                                    ]
                                }
                            ]
                        }
                    ],
                }
            ]
        }
    )
