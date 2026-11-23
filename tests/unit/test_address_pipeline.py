"""Correction attribution uses the existing stack and does not add rules."""

from __future__ import annotations

from docintel.services.address_pipeline import CORRECTION_VERSION, apply_existing_stack, word_diff


def test_character_fix_is_attributed_to_the_existing_layer():
    result = apply_existing_stack("P.0. BOX 487")
    assert result["corrected_text"] == "P.O. BOX 487"
    assert result["layers_applied"] == ["correct"]
    changed = [item for item in result["token_changes"] if item["change"] == "changed"]
    removed = [item for item in result["token_changes"] if item["change"] == "removed"]
    assert changed and changed[0]["layer"] == "correct"
    assert removed and removed[0]["layer"] == "correct"
    assert result["readers_ran"] is False


def test_leading_digit_is_attributed_to_boundary():
    result = apply_existing_stack("1 GCI")
    assert result["corrected_text"] == "GCI"
    removed = [item for item in result["token_changes"] if item["text"] == "1"]
    assert removed[0]["change"] == "removed"
    assert removed[0]["layer"] == "boundary"


def test_fallback_without_readers_does_not_invent_a_layer():
    result = apply_existing_stack("ACME STREET")
    assert result["corrected_text"] == "ACME STREET"
    assert result["layers_applied"] == []
    assert all(item["layer"] is None for item in result["token_changes"])


def test_demo_diff_has_no_correction_layer():
    tokens = word_diff("DEMO ORIGINAL VENDOR", "DEMO CORRECTED VENDOR")
    changed = [item for item in tokens if item["change"] != "unchanged"]
    assert changed
    assert all(item["layer"] is None for item in changed)
    assert CORRECTION_VERSION == "stack-v1:correct+boundary+fallback"
