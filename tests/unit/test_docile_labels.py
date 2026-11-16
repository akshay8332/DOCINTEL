"""Unit tests: DocILE word-to-field BIO assignment."""

from docintel.data.docile import label_words, layout_box


def test_layout_box_scales_unit_interval():
    assert layout_box([0.0, 0.0, 1.0, 0.5]) == [0, 0, 1000, 500]


def test_word_center_takes_smallest_containing_field():
    words = [
        ("990304502", [0.78, 0.34, 0.88, 0.36]),
        ("BIO", [0.07, 0.18, 0.20, 0.20]),
        ("Road", [0.17, 0.20, 0.22, 0.22]),
        ("other", [0.50, 0.50, 0.55, 0.52]),
    ]
    fields = [
        ("vendor_address", [0.05, 0.17, 0.47, 0.22]),
        ("vendor_name", [0.07, 0.18, 0.21, 0.20]),
        ("document_id", [0.78, 0.34, 0.89, 0.37]),
    ]
    assert label_words(words, fields) == [
        "B-document_id",
        "B-vendor_name",
        "B-vendor_address",
        "O",
    ]
