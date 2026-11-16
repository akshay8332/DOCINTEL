"""Multi-label targets, loss masking, alignment, and field grouping."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from multilabel import FIELD_NAMES
from multilabel.align import align_word_targets, word_probabilities
from multilabel.group import GROUPING_READING_ORDER, group_fields
from multilabel.ocr_text import correct_common_ocr
from multilabel.loss import masked_bce_with_logits
from multilabel.model import MultiLabelLayoutLM
from multilabel.targets import word_targets


class _Doc:
    def __init__(self, words, fields):
        self.document_id = "doc-p0"
        self.annotation_path = None
        self.pages = [type("P", (), {"words": words})()]
        self._fields = fields


def test_overlapping_amounts_stay_active_together(monkeypatch):
    def fields(_doc, _names=None):
        return [
            {"fieldtype": "amount_due", "text": "10.00", "bbox": [0.1, 0.1, 0.4, 0.2]},
            {"fieldtype": "amount_total_gross", "text": "10.00", "bbox": [0.1, 0.1, 0.2, 0.16]},
            {"fieldtype": "vendor_name", "text": "Acme", "bbox": [0.5, 0.5, 0.7, 0.6]},
            {"fieldtype": "vendor_address", "text": "Acme Road", "bbox": [0.48, 0.48, 0.8, 0.7]},
        ]

    monkeypatch.setattr("multilabel.targets.page_fields", fields)
    words = [
        type("W", (), {"text": "10.00", "box": [100, 100, 200, 160]})(),
        type("W", (), {"text": "Acme", "box": [500, 500, 600, 560]})(),
    ]
    result = word_targets(_Doc(words, None))
    assert result["matrix"][0] == [0, 0, 0, 0, 0, 0, 0, 1, 1]
    assert result["matrix"][1] == [0, 0, 0, 1, 0, 1, 0, 0, 0]
    assert result["unmatched_fields"] == []
    assert result["multi_label_words"] == 2


def test_unmatched_field_is_reported(monkeypatch):
    def fields(_doc, _names=None):
        return [{"fieldtype": "date_issue", "text": "8/03/20", "bbox": [0.9, 0.9, 0.95, 0.94]}]

    monkeypatch.setattr("multilabel.targets.page_fields", fields)
    words = [type("W", (), {"text": "other", "box": [0, 0, 10, 10]})()]
    result = word_targets(_Doc(words, None))
    assert result["matrix"] == [[0, 0, 0, 0, 0, 0, 0, 0, 0]]
    assert result["unmatched_fields"][0]["field"] == "date_issue"


def test_first_subword_keeps_every_active_label():
    word_rows = [
        [0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1],
    ]
    aligned = align_word_targets(word_rows, [None, 0, 0, 1, None], len(FIELD_NAMES))
    assert aligned["loss_mask"] == [0, 1, 0, 1, 0]
    assert aligned["labels"][1] == [0, 0, 0, 1, 0, 1, 0, 0, 0]
    assert aligned["labels"][2] == [0.0] * 9
    assert aligned["labels"][3][7] == 1 and aligned["labels"][3][8] == 1
    assert aligned["token_word"] == [-1, 0, -1, 1, -1]
    assert aligned["truncated_words"] == []


def test_ignored_tokens_do_not_change_the_loss():
    logits = torch.zeros(1, 3, 2)
    labels = torch.zeros(1, 3, 2)
    labels[0, 0, 0] = 1
    mask = torch.tensor([[1.0, 0.0, 0.0]])
    changed = labels.clone()
    changed[0, 1] = 1
    changed[0, 2] = 1
    first = masked_bce_with_logits(logits, labels, mask)
    second = masked_bce_with_logits(logits, changed, mask)
    assert torch.allclose(first, second)
    assert float(first) > 0


def test_sigmoid_head_returns_nine_logits():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"hidden_size": 4})()

        def forward(self, **_kwargs):
            batch, length = _kwargs["input_ids"].shape
            hidden = torch.zeros(batch, length, 4)
            return type("Out", (), {"last_hidden_state": hidden})()

    model = MultiLabelLayoutLM(Tiny(), num_labels=9)
    logits, loss = model(
        input_ids=torch.zeros(2, 5, dtype=torch.long),
        attention_mask=torch.ones(2, 5, dtype=torch.long),
        bbox=torch.zeros(2, 5, 4, dtype=torch.long),
        pixel_values=torch.zeros(2, 3, 8, 8),
        labels=torch.zeros(2, 5, 9),
        loss_mask=torch.tensor([[1, 0, 0, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.float32),
    )
    assert logits.shape == (2, 5, 9)
    assert torch.sigmoid(logits).shape == (2, 5, 9)
    assert loss is not None


def test_grouping_separates_instances_and_exports_relative_boxes():
    words = ["10.00", "note", "20.00", "far"]
    boxes = [[0, 0, 40, 10], [50, 0, 80, 10], [0, 20, 40, 30], [0, 200, 40, 210]]
    probabilities = [
        [0.9 if column == 7 else 0.0 for column in range(9)],
        [0.0] * 9,
        [0.8 if column == 7 else 0.0 for column in range(9)],
        [0.95 if column == 7 else 0.0 for column in range(9)],
    ]
    # The third amount is consecutive in the positive list but far down the page.
    probabilities[2][8] = 0.7
    fields = group_fields(words, boxes, probabilities, FIELD_NAMES, threshold=0.5, page_index=1)
    amounts = [field for field in fields if field.label == "amount_due"]
    gross = [field for field in fields if field.label == "amount_total_gross"]
    assert [field.word_indexes for field in amounts] == [(0,), (2,), (3,)]
    assert [field.text for field in amounts] == ["10.00", "20.00", "far"]
    assert gross[0].word_indexes == (2,)
    assert amounts[0].box_relative == [0.0, 0.0, 0.04, 0.01]
    assert amounts[0].page_index == 1
    exported = amounts[0].to_docile_dict()
    assert exported["fieldtype"] == "amount_due"
    assert exported["bbox"] == amounts[0].box_relative
    assert max(exported["bbox"]) <= 1


def test_address_lines_stay_one_block_and_keep_line_breaks():
    words = ["WHTH", "WNKO", "Proposal", "1000", "N.", "Street"]
    boxes = [
        [327, 30, 428, 52],
        [450, 29, 552, 54],
        [563, 31, 682, 57],
        [428, 57, 469, 71],
        [470, 58, 491, 72],
        [530, 57, 583, 71],
    ]
    probabilities = [[0.0] * 9 for _ in words]
    for index in (0, 1, 3, 4, 5):
        probabilities[index][5] = 0.9
    fields = group_fields(words, boxes, probabilities, FIELD_NAMES, threshold=0.5)
    addresses = [field for field in fields if field.label == "vendor_address"]
    assert len(addresses) == 1
    assert addresses[0].text == "WHTH WNKO\n1000 N. Street"
    assert addresses[0].word_indexes == (0, 1, 3, 4, 5)


def test_field_thresholds_can_differ():
    words = ["Tim", "Katz"]
    boxes = [[144, 149, 174, 164], [146, 199, 178, 211]]
    probabilities = [
        [0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.56, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.9, 0.0, 0.0],
    ]
    fields = group_fields(
        words,
        boxes,
        probabilities,
        FIELD_NAMES,
        field_thresholds=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.5, 0.5],
    )
    addresses = [field.text for field in fields if field.label == "customer_billing_address"]
    assert addresses == ["Katz"]


def test_common_ocr_fixes_do_not_invent_words():
    assert correct_common_ocr("P.0. BOX 487") == "P.O. BOX 487"
    assert correct_common_ocr("PA I5206-4490") == "PA 15206-4490"
    assert correct_common_ocr("WHTH WNKO Proposal") == "WHTH WNKO Proposal"


def test_reading_order_v1_splits_an_address_at_a_skipped_word():
    words = ["WHTH", "WNKO", "Proposal", "1000", "Street"]
    boxes = [
        [327, 30, 428, 52],
        [450, 29, 552, 54],
        [563, 31, 682, 57],
        [428, 57, 469, 71],
        [530, 57, 583, 71],
    ]
    probabilities = [[0.0] * 9 for _ in words]
    for index in (0, 1, 3, 4):
        probabilities[index][5] = 0.9
    fields = group_fields(
        words,
        boxes,
        probabilities,
        FIELD_NAMES,
        threshold=0.5,
        grouping=GROUPING_READING_ORDER,
    )
    addresses = [field for field in fields if field.label == "vendor_address"]
    assert [field.text for field in addresses] == ["WHTH WNKO", "1000 Street"]


def test_side_by_side_names_stay_separate():
    words = ["Cumulus Broadcasting LLC", "Cumulus Media Partners LLC"]
    boxes = [[320, 69, 442, 78], [473, 70, 603, 79]]
    probabilities = [
        [0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    fields = group_fields(words, boxes, probabilities, FIELD_NAMES, threshold=0.5)
    names = [field for field in fields if field.label == "vendor_name"]
    assert [field.text for field in names] == words


def test_one_word_can_emit_two_field_predictions():
    probabilities = [[0.0, 0.0, 0.0, 0.8, 0.0, 0.7, 0.0, 0.0, 0.0]]
    fields = group_fields(
        ["Acme"],
        [[10, 10, 40, 20]],
        probabilities,
        FIELD_NAMES,
        threshold=0.5,
    )
    assert {field.label for field in fields} == {"vendor_name", "vendor_address"}
    assert fields[0].word_indexes == fields[1].word_indexes == (0,)


def test_word_probabilities_follow_the_first_subword():
    token_probabilities = [
        [0.1] * 9,
        [0.9] * 9,
        [0.2] * 9,
        [0.4] * 9,
    ]
    words = word_probabilities(token_probabilities, [None, 0, 0, 1], 2)
    assert words[0][0] == 0.9
    assert words[1][0] == 0.4
