"""Selective prediction-text edits. No checkpoint is loaded."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from multilabel.post_ocr import (
    correct_prediction,
    normalize_prediction,
    refine_boundary,
    same_wording,
    selective_fallback,
    stack_correction,
    word_change_outcomes,
)


def test_normalization_keeps_the_wording():
    original = "NEW YORK , NY 10010-5201"
    edited = normalize_prediction(original)
    assert edited == "NEW YORK, NY 10010-5201"
    assert same_wording(original, edited)


def test_character_fixes_cover_po_box_and_care_of():
    assert correct_prediction("P.0. Box 487\nclo National City Bank") == "P.O. Box 487\nc/o National City Bank"
    assert correct_prediction("WHTH WNKO Proposal") == "WHTH WNKO Proposal"


def test_boundary_drops_a_repeated_line_and_a_leading_digit():
    assert refine_boundary("1 GCI\n92 GRANDVIEW AVENUE") == "GCI\n92 GRANDVIEW AVENUE"
    assert refine_boundary("BSMG Worldwide, Inc.\nP.O. Box 100583") == "BSMG Worldwide, Inc.\nP.O. Box 100583"
    assert refine_boundary("SAUL\nSAUL\n1500 MARKET STREET") == "SAUL\n1500 MARKET STREET"


def test_fallback_replaces_a_token_only_when_both_readers_agree():
    prediction = "Tallahassce, FL 32310"
    agreed = "Tallahassee, FL 32310"
    assert selective_fallback(prediction, [agreed, agreed]) == agreed
    assert selective_fallback(prediction, [agreed, prediction]) == prediction


def test_fallback_does_not_adopt_split_punctuation_from_one_reader():
    prediction = "P.O. Box 1057"
    fragmented = "P . O . Box 1057"
    assert selective_fallback(prediction, [prediction, fragmented]) == prediction


def test_stack_keeps_a_company_line():
    original = "BSMG Worldwide, Inc.\nP.O. Box 100583"
    assert stack_correction(original, [original, original]) == original


def test_a_changed_word_is_marked_when_it_moves_toward_or_away_from_gold():
    assert word_change_outcomes("P.0. BOX 487", "P.O. BOX 487", "P.O. BOX 487") == ["improved"]
    assert word_change_outcomes("P.O. BOX 487", "P.0. BOX 487", "P.O. BOX 487") == ["harmed"]
    assert word_change_outcomes("Tallahassee, FL", "Tallahassce, FL", "Portland, OR") == ["neutral"]
