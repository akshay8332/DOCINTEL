"""Root-cause labels for address text. No checkpoint is loaded."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from multilabel.address_text_audit import build_report, classify_address_field, render_summary
from multilabel.checkpoint import assert_diagnostic_destination

INSIDE = [10, 10, 80, 40]
INSIDE_2 = [90, 10, 160, 40]
OUTSIDE = [800, 800, 900, 900]
BOX = [0.0, 0.0, 0.5, 0.2]


def _word(text, box, probability, truncated=False):
    return {"text": text, "box": box, "probability": probability, "truncated": truncated}


def _gold(text, field="vendor_address"):
    return {
        "document_id": "doc",
        "page": 0,
        "fieldtype": field,
        "text": text,
        "bbox": BOX,
    }


def _prediction(text, indexes, bbox=None):
    return {
        "fieldtype": "vendor_address",
        "text": text,
        "bbox": BOX if bbox is None else bbox,
        "score": 0.9,
        "word_indexes": indexes,
        "page": 0,
    }


def test_exact_text_is_not_a_mismatch():
    words = [_word("1", INSIDE, 0.9), _word("Main", INSIDE_2, 0.9)]
    case = classify_address_field(_gold("1 Main"), words, [_prediction("1 Main", (0, 1))])
    assert case["exact"] is True
    assert case["primary"] == "exact"


def test_missing_ocr_word():
    words = [_word("1", INSIDE, 0.9), _word("Main", INSIDE_2, 0.9)]
    case = classify_address_field(_gold("1 Main Street"), words, [_prediction("1 Main", (0, 1))])
    assert case["primary"] == "missing_ocr"
    assert case["missing_ocr"] == [{"gold": "Street", "on_page": False}]


def test_ocr_character_error():
    words = [_word("Streat", INSIDE, 0.9)]
    case = classify_address_field(_gold("Street"), words, [_prediction("Streat", (0,))])
    assert case["primary"] == "ocr_character"
    assert case["ocr_character"][0]["gold"] == "Street"


def test_threshold_excludes_a_gold_word():
    words = [_word("Street", INSIDE, 0.2)]
    case = classify_address_field(_gold("Street"), words, [])
    assert case["primary"] == "threshold"
    assert case["threshold"][0]["text"] == "Street"
    assert case["prediction"] is None


def test_unwanted_word_outside_the_gold_box():
    words = [_word("1", INSIDE, 0.9), _word("Phone", OUTSIDE, 0.8)]
    case = classify_address_field(_gold("1"), words, [_prediction("1 Phone", (0, 1))])
    assert case["primary"] == "unwanted_nearby"
    assert case["unwanted_nearby"][0]["text"] == "Phone"
    assert case["unwanted_nearby"][0]["inside_gold_box"] is False


def test_newline_difference_is_line_grouping():
    words = [_word("1", INSIDE, 0.9), _word("Main", INSIDE_2, 0.9)]
    case = classify_address_field(_gold("1\nMain"), words, [_prediction("1 Main", (0, 1))])
    assert case["primary"] == "line_grouping"
    assert case["line_grouping"]["whitespace_only"] is True


def test_same_words_in_another_order():
    words = [_word("Main", INSIDE, 0.9), _word("Street", INSIDE_2, 0.9)]
    case = classify_address_field(_gold("Main Street"), words, [_prediction("Street Main", (0, 1))])
    assert case["primary"] == "word_order"


def test_report_keeps_examples_and_summary():
    words = [_word("Streat", INSIDE, 0.9)]
    case = classify_address_field(_gold("Street"), words, [_prediction("Streat", (0,))])
    report = build_report([case], checkpoint="epoch3", pages=1, documents=1)
    assert report["fields"]["vendor_address"]["primary_cause"]["ocr_character"] == 1
    assert report["examples"]["vendor_address"]["ocr_character"][0]["prediction"] == "Streat"
    summary = render_summary(report)
    assert "ocr_character" in summary
    assert "Streat" in summary
    assert report["checkpoint_updated"] is False


def test_audit_uses_a_new_filename_and_leaves_old_reports_alone():
    assert_diagnostic_destination(ROOT / "artifacts" / "diagnostics" / "address-text-audit-100.json")
    try:
        assert_diagnostic_destination(
            ROOT / "artifacts" / "diagnostics" / "multilabel-reconstruction-100.json"
        )
    except RuntimeError as exc:
        assert "multilabel-reconstruction-100.json" in str(exc)
    else:
        raise AssertionError("existing reconstruction report must stay frozen")
