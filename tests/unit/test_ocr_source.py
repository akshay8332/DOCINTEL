"""Word-box scaling and DocILE identity checks. docTR is not launched."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docintel.services.address_pipeline import InferenceUnavailable
from docintel.services.ocr_source import doctr_interpreter, matching_docile_document, scale_relative_box


def test_relative_box_uses_the_training_scale():
    assert scale_relative_box([0.0, 0.0, 0.5, 1.0]) == [0, 0, 500, 1000]
    assert scale_relative_box([0.2, 0.2, 0.2, 0.4])[2] == 201


def test_docile_ocr_is_reused_only_when_the_bytes_match(tmp_path: Path):
    doc_id = "0123456789abcdef01234567"
    data = tmp_path / "docile"
    (data / "pdfs").mkdir(parents=True)
    (data / "ocr").mkdir()
    pdf = data / "pdfs" / f"{doc_id}.pdf"
    pdf.write_bytes(b"%PDF-1.4\nmatch\n")
    (data / "ocr" / f"{doc_id}.json").write_text("{}", encoding="utf-8")
    (data / "val.json").write_text(json.dumps([doc_id]), encoding="utf-8")

    upload = tmp_path / "upload.pdf"
    upload.write_bytes(pdf.read_bytes())
    assert matching_docile_document(f"{doc_id}.pdf", upload, data) == doc_id

    upload.write_bytes(b"%PDF-1.4\ndifferent\n")
    assert matching_docile_document(f"{doc_id}.pdf", upload, data) is None
    assert matching_docile_document("invoice.pdf", pdf, data) is None


def test_doctr_interpreter_uses_the_configured_python(tmp_path: Path, monkeypatch):
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    monkeypatch.setenv("DOCINTEL_DOCTR_PYTHON", str(python))
    assert doctr_interpreter() == python


def test_doctr_interpreter_reports_when_no_install_exists(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DOCINTEL_DOCTR_PYTHON", str(tmp_path / "missing"))
    monkeypatch.setattr("docintel.services.ocr_source.repo_root", lambda: tmp_path)
    with pytest.raises(InferenceUnavailable):
        doctr_interpreter()
