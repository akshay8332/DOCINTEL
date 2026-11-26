"""Invoice API with an explicit demo pipeline. Real mode fails closed without the checkpoint."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from docintel.api.main import create_app
from docintel.schemas.config import AppSettings
from docintel.services.invoice_service import safe_filename
from fastapi.testclient import TestClient

PDF = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"


def test_safe_filename_strips_windows_and_posix_paths():
    assert safe_filename(r"..\secret\invoice.pdf") == "invoice.pdf"
    assert safe_filename("../secret/invoice.pdf") == "invoice.pdf"
    assert safe_filename("invoice.pdf") == "invoice.pdf"


def _app(tmp_path: Path, mode: str = "mock"):
    settings = AppSettings(
        storage_root=tmp_path / "uploads",
        checkpoint_dir=tmp_path / "checkpoints",
        invoice_db_url="sqlite:///" + (tmp_path / "app" / "invoices.db").as_posix(),
        inference_mode=mode,
        process_inline=True,
        max_upload_bytes=64,
    )
    return TestClient(create_app(settings))


def _upload(client: TestClient, content: bytes = PDF, name: str = "invoice.pdf"):
    return client.post("/api/invoices/upload", files={"files": (name, content, "application/pdf")})


def test_health_reports_demo_mode(tmp_path: Path):
    client = _app(tmp_path)
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["demo"] is True
    assert payload["inference_mode"] == "mock"
    assert payload["threshold"] == 0.5
    assert payload["grouping"] == "line-block-v2"
    assert "61" not in str(payload)
    assert "accuracy" not in payload


def test_upload_list_and_reject_invalid_files(tmp_path: Path):
    client = _app(tmp_path)
    empty = client.get("/api/invoices").json()
    assert empty["summary"]["total"] == 0
    assert empty["invoices"] == []

    bad = _upload(client, b"not a pdf", "notes.pdf")
    assert bad.status_code == 400

    huge = _upload(client, b"%PDF" + b"x" * 80, "big.pdf")
    assert huge.status_code == 400

    created = _upload(client, PDF, r"..\secret\invoice.pdf")
    assert created.status_code == 200
    invoice = created.json()["accepted"][0]
    assert invoice["filename"] == "invoice.pdf"
    assert "file_path" not in invoice
    assert "stored_name" not in invoice
    stored = list((tmp_path / "uploads").glob("*.pdf"))
    assert len(stored) == 1
    assert stored[0].parent == (tmp_path / "uploads").resolve()

    listing = client.get("/api/invoices").json()
    assert listing["summary"]["total"] == 1
    assert listing["summary"]["processed"] == 0


def test_demo_processing_persists_original_and_corrected(tmp_path: Path):
    client = _app(tmp_path)
    invoice_id = _upload(client).json()["accepted"][0]["id"]
    processed = client.post(f"/api/invoices/{invoice_id}/process")
    assert processed.status_code == 200
    assert processed.json()["processing_status"] == "completed"

    results = client.get(f"/api/invoices/{invoice_id}/results").json()
    assert results["run"]["mode"] == "mock"
    assert results["run"]["checkpoint_reference"] is None
    names = {field["field_name"] for field in results["fields"]}
    assert names == {"vendor_address", "customer_billing_address"}
    for field in results["fields"]:
        assert field["demo"] is True
        assert field["original_text"]
        assert field["corrected_text"]
        assert field["original_text"] != field["corrected_text"]
        assert field["available_confidence_signals"]["model_score"] is None
        assert field["available_confidence_signals"]["calibrated_probability"] is None
        assert "file_path" not in field
        assert all(token["layer"] is None for token in field["token_changes"])


def test_review_approve_and_reject_keep_the_original(tmp_path: Path):
    client = _app(tmp_path)
    invoice_id = _upload(client).json()["accepted"][0]["id"]
    client.post(f"/api/invoices/{invoice_id}/process")
    before = client.get(f"/api/invoices/{invoice_id}/results").json()
    vendor = next(field for field in before["fields"] if field["field_name"] == "vendor_address")

    edited = client.put(
        f"/api/invoices/{invoice_id}/review",
        json={
            "field_name": "vendor_address",
            "approved_value": "HUMAN VENDOR ADDRESS",
            "reviewer_comment": "Checked the page.",
            "correction_reason": "Reviewer typed the address.",
        },
    )
    assert edited.status_code == 200
    assert edited.json()["original_text"] == vendor["original_text"]
    assert edited.json()["corrected_text"] == vendor["corrected_text"]
    assert edited.json()["approved_value"] == "HUMAN VENDOR ADDRESS"

    approved = client.post(f"/api/invoices/{invoice_id}/approve", json={"reviewer_comment": "Accepted."})
    assert approved.status_code == 200
    assert approved.json()["review_status"] == "approved"
    after = client.get(f"/api/invoices/{invoice_id}/results").json()
    vendor_after = next(field for field in after["fields"] if field["field_name"] == "vendor_address")
    customer = next(field for field in after["fields"] if field["field_name"] == "customer_billing_address")
    assert vendor_after["original_text"] == vendor["original_text"]
    assert vendor_after["approved_value"] == "HUMAN VENDOR ADDRESS"
    assert customer["approved_value"] == customer["corrected_text"]

    rejected = client.post(f"/api/invoices/{invoice_id}/reject", json={"reviewer_comment": "Wrong document."})
    assert rejected.json()["review_status"] == "rejected"
    final = client.get(f"/api/invoices/{invoice_id}/results").json()
    vendor_final = next(field for field in final["fields"] if field["field_name"] == "vendor_address")
    assert vendor_final["original_text"] == vendor["original_text"]
    assert vendor_final["corrected_text"] == vendor["corrected_text"]

    history = client.get(f"/api/invoices/{invoice_id}/history").json()
    actions = [item["reviewer_action"] for item in history["reviews"]]
    assert actions.count("edited") == 1
    assert "approved" in actions
    assert "rejected" in actions
    assert history["runs"][0]["correction_version"] == "stack-v1:correct+boundary+fallback"


def test_real_mode_without_the_frozen_checkpoint_fails(tmp_path: Path):
    client = _app(tmp_path, mode="real")
    health = client.get("/api/health").json()
    assert health["demo"] is False
    assert health["checkpoint_present"] is False
    assert health["inference_mode"] == "unavailable"

    invoice_id = _upload(client).json()["accepted"][0]["id"]
    processed = client.post(f"/api/invoices/{invoice_id}/process")
    assert processed.json()["processing_status"] == "failed"
    message = processed.json()["error_message"]
    assert "not installed" in message
    assert "docile_multilabel_200" not in message
    assert ":\\" not in message

    results = client.get(f"/api/invoices/{invoice_id}/results").json()
    assert results["fields"] == []
    listing = client.get("/api/invoices").json()
    assert listing["summary"]["failed"] == 1
    assert "accuracy" not in listing["summary"]


def test_second_run_is_kept_and_export_is_labeled(tmp_path: Path):
    client = _app(tmp_path)
    invoice_id = _upload(client).json()["accepted"][0]["id"]
    client.post(f"/api/invoices/{invoice_id}/process")
    client.post(f"/api/invoices/{invoice_id}/process")
    history = client.get(f"/api/invoices/{invoice_id}/history").json()
    assert len(history["runs"]) == 2

    exported = client.get(f"/api/invoices/{invoice_id}/export", params={"format": "csv"})
    assert exported.status_code == 200
    assert "original_text" in exported.text
    assert "DEMO ORIGINAL VENDOR" in exported.text

    compared = client.post(
        f"/api/invoices/{invoice_id}/compare",
        json={"vendor_address": "DEMO CORRECTED VENDOR\n100 MAIN STREET", "customer_billing_address": ""},
    )
    body = compared.json()
    assert "not a model accuracy score" in body["note"]
    vendor = next(item for item in body["fields"] if item["field_name"] == "vendor_address")
    customer = next(item for item in body["fields"] if item["field_name"] == "customer_billing_address")
    assert vendor["corrected_exact"] is True
    assert customer["reference_supplied"] is False
    assert customer["corrected_exact"] is None
