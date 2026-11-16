"""Phase 2 decoder, overlap labels, and DocILE box conversion."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "packages"))

from docintel.models.layoutlm import _spans_from_bio
from phase2.bio_decode import (
    count_bio_by_field,
    decode_bio_labels,
    first_subword_agreement,
    mapping_errors,
)
from phase2.multilabel import (
    IndependentSigmoidHead,
    annotation_retention,
    exclusive_name,
    field_scores,
    multilabel_matrix,
)
from phase2.ocr_compare import compare_ocr, merge_page_reports
from phase2.spans import spans_from_word_labels, to_relative_box


def test_decoder_follows_each_bio_rule():
    labels = [
        "O",
        "I-vendor_name",
        "I-vendor_name",
        "O",
        "B-date_issue",
        "I-date_issue",
        "I-vendor_address",
    ]
    spans = decode_bio_labels(labels)
    assert [(span.label, span.start, span.end, span.recovered) for span in spans] == [
        ("vendor_name", 1, 3, True),
        ("date_issue", 4, 6, False),
        ("vendor_address", 6, 7, True),
    ]


def test_existing_decoder_drops_a_missing_b_tag():
    words = ["Acme", "Inc"]
    labels = ["I-vendor_name", "I-vendor_name"]
    boxes = [[0, 0, 10, 10], [12, 0, 20, 10]]
    assert _spans_from_bio(words, labels, [0.5, 0.5], boxes) == []
    recovered = spans_from_word_labels(words, labels, boxes, [0.4, 0.6], page_index=2)
    assert len(recovered) == 1
    assert recovered[0].recovered is True
    assert recovered[0].text == "Acme Inc"
    assert recovered[0].box_0_1000 == [0, 0, 20, 10]
    assert recovered[0].page_index == 2


def test_relative_box_uses_the_unit_interval():
    assert to_relative_box([410, 57, 600, 104]) == [0.41, 0.057, 0.6, 0.104]


def test_b_tags_stay_on_the_first_subword():
    report = first_subword_agreement(
        ["B-document_id", "O"],
        [None, 0, 0, 1, None],
        ["O", "B-document_id", "I-document_id", "O", "O"],
    )
    assert report["b_kept_on_first_subword"] == 1
    assert report["b_changed_on_first_subword"] == 0
    assert report["b_truncated"] == 0


def test_checkpoint_map_must_match_label_order():
    labels = ["O", "B-document_id", "I-document_id"]
    assert mapping_errors({0: "O", 1: "B-document_id", 2: "I-document_id"}, labels) == []
    assert mapping_errors({"0": "O", "1": "I-document_id", "2": "B-document_id"}, labels)


def test_overlapping_amount_is_kept_for_both_fields():
    words = [("5020.24", [0.10, 0.10, 0.20, 0.12])]
    fields = [
        ("amount_due", [0.09, 0.09, 0.30, 0.16]),
        ("amount_total_gross", [0.10, 0.10, 0.20, 0.12]),
    ]
    names = ["amount_due", "amount_total_gross"]
    assert exclusive_name(words, fields, allowed=set(names)) == ["amount_total_gross"]
    assert multilabel_matrix(words, fields, names) == [[1, 1]]
    retention = annotation_retention(words, fields, allowed=set(names))
    assert retention["amount_due"] == {"annotations": 1, "exclusive": 0, "multi": 1}
    assert retention["amount_total_gross"]["exclusive"] == 1


def test_sigmoid_head_accepts_two_labels_on_one_token():
    head = IndependentSigmoidHead(hidden_size=4, num_labels=2)
    hidden = torch.zeros(1, 3, 4)
    labels = torch.tensor([[[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]])
    mask = torch.tensor([[1, 1, 0]])
    logits, loss = head(hidden, labels, mask)
    assert logits.shape == (1, 3, 2)
    assert loss is not None and float(loss.detach()) >= 0
    scores = field_scores(
        [[1, 1], [0, 1]],
        [[1, 0], [0, 1]],
        ["amount_due", "amount_total_gross"],
    )
    assert scores["amount_due"]["f1"] == 1.0
    assert scores["amount_total_gross"]["recall"] == 0.5


def test_ocr_digit_match_counts_a_truncated_amount_as_wrong():
    fields = [
        {"fieldtype": "amount_due", "text": "$806.00", "bbox": [0.1, 0.1, 0.3, 0.2]},
        {"fieldtype": "date_issue", "text": "03/25/99", "bbox": [0.5, 0.5, 0.7, 0.6]},
    ]
    words = [
        {"text": "$806.", "box": [0.12, 0.12, 0.28, 0.18]},
        {"text": "03/25/99", "box": [0.52, 0.52, 0.68, 0.58]},
    ]
    report = compare_ocr(
        fields,
        words,
        amount_types={"amount_due"},
        date_types={"date_issue"},
    )
    assert report["detected_fields"] == 2
    assert report["amount_digit_hits"] == 0
    assert report["date_digit_hits"] == 1
    pooled = merge_page_reports([report, report])
    assert pooled["fields"] == 4
    assert pooled["date_digit_hits"] == 2


def test_bio_counts_separate_each_field():
    counts = count_bio_by_field(
        ["O", "B-document_id", "B-vendor_address", "I-vendor_address"],
        ["document_id", "vendor_address"],
    )
    assert counts["document_id"] == {"B": 1, "I": 0}
    assert counts["vendor_address"] == {"B": 1, "I": 1}
    assert counts["O"]["O"] == 1
