"""Upload, background processing, and review. Original predictions stay immutable."""

from __future__ import annotations

import csv
import io
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from docintel.database.models import ExtractionResult, Invoice, ProcessingRun, ReviewHistory
from docintel.schemas.config import AppSettings
from docintel.services.address_pipeline import (
    ADDRESS_FIELDS,
    CORRECTION_VERSION,
    AddressPipeline,
    InferenceUnavailable,
)

_NAME = re.compile(r"[^A-Za-z0-9._ -]+")


def safe_filename(name: str) -> str:
    # Strip directory parts for both Windows and POSIX separators. Path.name alone
    # keeps backslashes on Linux, which would leave names like "_secret_invoice.pdf".
    base = (name or "invoice.pdf").replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = _NAME.sub("_", base).strip(" .") or "invoice.pdf"
    if not cleaned.lower().endswith(".pdf"):
        cleaned += ".pdf"
    return cleaned[:180]


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def public_invoice(row: Invoice) -> dict:
    return {
        "id": row.id,
        "filename": row.filename,
        "uploaded_at": _iso(row.uploaded_at),
        "processing_status": row.processing_status,
        "review_status": row.review_status,
        "page_count": row.page_count,
        "error_message": row.error_message,
    }


def public_field(row: ExtractionResult) -> dict:
    return {
        "id": row.id,
        "field_name": row.field_name,
        "original_text": row.original_text,
        "corrected_text": row.corrected_text,
        "approved_value": row.approved_value,
        "page_number": row.page_number,
        "page_index": row.page_index,
        "bounding_box": row.bounding_box,
        "box_0_1000": row.box_0_1000,
        "available_confidence_signals": row.available_confidence_signals,
        "token_changes": row.token_changes,
        "layers_applied": row.layers_applied,
        "review_status": row.review_status,
        "demo": bool(row.demo),
    }


def public_run(row: ProcessingRun, *, with_fields: bool = False) -> dict:
    payload = {
        "id": row.id,
        "invoice_id": row.invoice_id,
        "model_version": row.model_version,
        "checkpoint_reference": row.checkpoint_reference,
        "correction_version": row.correction_version,
        "configuration": row.configuration,
        "processing_duration": row.processing_duration,
        "created_at": _iso(row.created_at),
        "error_message": row.error_message,
        "mode": row.mode,
    }
    if with_fields:
        payload["fields"] = [public_field(item) for item in row.fields]
    return payload


def public_review(row: ReviewHistory) -> dict:
    return {
        "id": row.id,
        "extraction_result_id": row.extraction_result_id,
        "original_prediction": row.original_prediction,
        "corrected_prediction": row.corrected_prediction,
        "approved_value": row.approved_value,
        "reviewer_action": row.reviewer_action,
        "reviewer_comment": row.reviewer_comment,
        "correction_reason": row.correction_reason,
        "model_version": row.model_version,
        "correction_version": row.correction_version,
        "created_at": _iso(row.created_at),
    }


class InvoiceService:
    def __init__(self, session_factory, settings: AppSettings, pipeline: AddressPipeline) -> None:
        self.session_factory = session_factory
        self.settings = settings
        self.pipeline = pipeline
        self.upload_root = settings.storage_root.resolve()
        self.upload_root.mkdir(parents=True, exist_ok=True)

    def session(self) -> Session:
        return self.session_factory()

    def summary(self, db: Session) -> dict:
        rows = list(db.scalars(select(Invoice)))
        return {
            "total": len(rows),
            "processed": sum(item.processing_status == "completed" for item in rows),
            "pending_review": sum(
                item.processing_status == "completed" and item.review_status == "pending" for item in rows
            ),
            "approved": sum(item.review_status == "approved" for item in rows),
            "failed": sum(item.processing_status == "failed" for item in rows),
        }

    def list_invoices(self, db: Session) -> list[Invoice]:
        return list(db.scalars(select(Invoice).order_by(Invoice.uploaded_at.desc())))

    def get_invoice(self, db: Session, invoice_id: str) -> Invoice | None:
        return db.get(Invoice, invoice_id)

    def stored_file(self, row: Invoice) -> Path:
        path = (self.upload_root / row.stored_name).resolve()
        if path.parent != self.upload_root or not path.is_file():
            raise FileNotFoundError(row.id)
        return path

    def accept_upload(self, db: Session, filename: str, content: bytes) -> Invoice:
        if len(content) > self.settings.max_upload_bytes:
            limit = self.settings.max_upload_bytes
            label = f"{limit // (1024 * 1024)} MB" if limit >= 1024 * 1024 else f"{limit} bytes"
            raise ValueError(f"This file is larger than the {label} limit.")
        if len(content) < 5 or not content.startswith(b"%PDF"):
            raise ValueError("Only PDF files are accepted.")
        invoice_id = str(uuid.uuid4())
        stored = f"{invoice_id}.pdf"
        destination = self.upload_root / stored
        destination.write_bytes(content)
        row = Invoice(
            id=invoice_id,
            filename=safe_filename(filename),
            stored_name=stored,
            processing_status="uploaded",
            review_status="pending",
            page_count=_page_count(content),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row

    def start_processing(self, invoice_id: str) -> None:
        if self.settings.process_inline:
            self.run_job(invoice_id)
            return
        threading.Thread(target=self.run_job, args=(invoice_id,), daemon=True).start()

    def mark_processing(self, db: Session, invoice_id: str) -> Invoice:
        row = self._require(db, invoice_id)
        if row.processing_status in {"processing", "rendered", "ocr", "inference", "corrections"}:
            raise RuntimeError("This invoice is already processing.")
        row.processing_status = "processing"
        row.error_message = None
        db.commit()
        db.refresh(row)
        return row

    def run_job(self, invoice_id: str) -> None:
        db = self.session()
        try:
            row = self._require(db, invoice_id)
            path = self.stored_file(row)

            def on_stage(stage: str) -> None:
                row.processing_status = stage
                db.commit()

            try:
                outcome = self.pipeline.process_invoice(path, filename=row.filename, on_stage=on_stage)
            except InferenceUnavailable as exc:
                self._fail(db, row, str(exc))
                return
            except Exception as exc:  # noqa: BLE001 - record the failure, do not invent fields
                self._fail(db, row, _public_error(exc))
                return
            run = ProcessingRun(
                id=str(uuid.uuid4()),
                invoice_id=row.id,
                model_version=outcome["model_version"],
                checkpoint_reference=outcome["checkpoint_reference"],
                correction_version=outcome["correction_version"],
                configuration=outcome["configuration"],
                processing_duration=outcome["duration_seconds"],
                mode=outcome["mode"],
            )
            db.add(run)
            for field in outcome["fields"]:
                db.add(
                    ExtractionResult(
                        id=str(uuid.uuid4()),
                        processing_run_id=run.id,
                        field_name=field["field_name"],
                        original_text=field["original_text"],
                        corrected_text=field["corrected_text"],
                        page_number=field["page_number"],
                        page_index=field["page_index"],
                        bounding_box=field["bounding_box"],
                        box_0_1000=field["box_0_1000"],
                        available_confidence_signals=field["confidence_signals"],
                        token_changes=field["token_changes"],
                        layers_applied=field["layers_applied"],
                        review_status="pending",
                        demo=1 if field.get("demo") else 0,
                    )
                )
            row.processing_status = "completed"
            row.review_status = "pending"
            row.error_message = None
            if outcome.get("page_count"):
                row.page_count = outcome["page_count"]
            db.commit()
        finally:
            db.close()

    def latest_run(self, db: Session, invoice_id: str) -> ProcessingRun | None:
        statement = (
            select(ProcessingRun)
            .where(ProcessingRun.invoice_id == invoice_id)
            .order_by(ProcessingRun.created_at.desc())
            .options(selectinload(ProcessingRun.fields))
        )
        return db.scalars(statement).first()

    def history(self, db: Session, invoice_id: str) -> dict:
        self._require(db, invoice_id)
        runs = list(
            db.scalars(
                select(ProcessingRun)
                .where(ProcessingRun.invoice_id == invoice_id)
                .order_by(ProcessingRun.created_at.desc())
                .options(selectinload(ProcessingRun.fields).selectinload(ExtractionResult.reviews))
            )
        )
        reviews = [public_review(item) for run in runs for field in run.fields for item in field.reviews]
        reviews.sort(key=lambda item: item["created_at"] or "")
        return {
            "invoice_id": invoice_id,
            "runs": [public_run(run, with_fields=True) for run in runs],
            "reviews": reviews,
        }

    def save_edit(
        self,
        db: Session,
        invoice_id: str,
        field_name: str,
        approved_value: str,
        comment: str,
        reason: str,
    ) -> ExtractionResult:
        if field_name not in ADDRESS_FIELDS:
            raise ValueError("Only the vendor address and the customer billing address can be reviewed.")
        result, run = self._latest_field(db, invoice_id, field_name)
        original = result.original_text
        corrected = result.corrected_text
        self._record(db, result, run, "edited", approved_value, comment, reason)
        result.approved_value = approved_value
        result.review_status = "edited"
        if result.original_text != original or result.corrected_text != corrected:
            raise RuntimeError("A review tried to change the stored prediction.")
        invoice = self._require(db, invoice_id)
        invoice.review_status = "pending"
        db.commit()
        db.refresh(result)
        return result

    def decide(self, db: Session, invoice_id: str, action: str, comment: str) -> Invoice:
        invoice = self._require(db, invoice_id)
        run = self.latest_run(db, invoice_id)
        if run is None or invoice.processing_status != "completed":
            raise RuntimeError("Process the invoice before reviewing it.")
        for result in run.fields:
            if action == "approved":
                value = result.approved_value if result.approved_value is not None else result.corrected_text
                result.approved_value = value
                result.review_status = "approved"
            else:
                value = result.approved_value
                result.review_status = "rejected"
            self._record(db, result, run, action, value, comment, "")
        invoice.review_status = action
        db.commit()
        db.refresh(invoice)
        return invoice

    def export_csv(self, db: Session, invoice_id: str) -> str:
        run = self.latest_run(db, invoice_id)
        if run is None:
            raise RuntimeError("This invoice has no extraction results.")
        buffer = io.StringIO()
        writer = csv.DictWriter(
            buffer,
            fieldnames=[
                "field_name",
                "original_text",
                "corrected_text",
                "approved_value",
                "page_number",
                "review_status",
                "model_version",
                "correction_version",
                "demo",
            ],
        )
        writer.writeheader()
        for field in run.fields:
            writer.writerow(
                {
                    "field_name": field.field_name,
                    "original_text": field.original_text,
                    "corrected_text": field.corrected_text,
                    "approved_value": field.approved_value or "",
                    "page_number": field.page_number if field.page_number is not None else "",
                    "review_status": field.review_status,
                    "model_version": run.model_version or "",
                    "correction_version": run.correction_version,
                    "demo": bool(field.demo),
                }
            )
        return buffer.getvalue()

    def compare_reference(self, db: Session, invoice_id: str, references: dict[str, str]) -> dict:
        run = self.latest_run(db, invoice_id)
        if run is None:
            raise RuntimeError("This invoice has no extraction results.")
        supplied = {key: value for key, value in references.items() if key in ADDRESS_FIELDS and value.strip()}
        comparisons = []
        for field in run.fields:
            reference = supplied.get(field.field_name)
            item = {
                "field_name": field.field_name,
                "reference_supplied": reference is not None,
                "original_exact": None,
                "corrected_exact": None,
                "approved_exact": None,
            }
            if reference is not None:
                item["original_exact"] = field.original_text.strip() == reference.strip()
                item["corrected_exact"] = field.corrected_text.strip() == reference.strip()
                if field.approved_value is not None:
                    item["approved_exact"] = field.approved_value.strip() == reference.strip()
            comparisons.append(item)
        return {
            "invoice_id": invoice_id,
            "note": "This compares the prediction with text you supplied. It is not a model accuracy score.",
            "fields": comparisons,
        }

    def _latest_field(self, db: Session, invoice_id: str, field_name: str) -> tuple[ExtractionResult, ProcessingRun]:
        run = self.latest_run(db, invoice_id)
        if run is None:
            raise RuntimeError("Process the invoice before reviewing it.")
        for result in run.fields:
            if result.field_name == field_name:
                return result, run
        raise RuntimeError("That field is not on the latest processing run.")

    def _record(
        self,
        db: Session,
        result: ExtractionResult,
        run: ProcessingRun,
        action: str,
        approved_value: str | None,
        comment: str,
        reason: str,
    ) -> None:
        db.add(
            ReviewHistory(
                id=str(uuid.uuid4()),
                extraction_result_id=result.id,
                original_prediction=result.original_text,
                corrected_prediction=result.corrected_text,
                approved_value=approved_value,
                reviewer_action=action,
                reviewer_comment=comment or "",
                correction_reason=reason or "",
                model_version=run.model_version,
                correction_version=run.correction_version or CORRECTION_VERSION,
            )
        )

    def _fail(self, db: Session, row: Invoice, message: str) -> None:
        run = ProcessingRun(
            id=str(uuid.uuid4()),
            invoice_id=row.id,
            model_version=None,
            checkpoint_reference=None,
            correction_version=CORRECTION_VERSION,
            configuration={"mode": "unavailable", "threshold": 0.5, "grouping": "line-block-v2"},
            processing_duration=0,
            error_message=message,
            mode="unavailable",
        )
        db.add(run)
        if row.processing_status not in {"rendered", "ocr", "inference", "corrections"}:
            row.processing_status = "failed"
        row.error_message = message
        db.commit()

    def _require(self, db: Session, invoice_id: str) -> Invoice:
        row = self.get_invoice(db, invoice_id)
        if row is None:
            raise KeyError(invoice_id)
        return row


def _page_count(content: bytes) -> int | None:
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return None
    import os
    import tempfile

    descriptor, name = tempfile.mkstemp(suffix=".pdf")
    os.close(descriptor)
    path = Path(name)
    try:
        path.write_bytes(content)
        pdf = pdfium.PdfDocument(str(path))
        try:
            return len(pdf)
        finally:
            pdf.close()
    except (OSError, RuntimeError, ValueError):
        return None
    finally:
        path.unlink(missing_ok=True)


def _public_error(exc: Exception) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    if ":\\" in text or text.startswith("/") or "Traceback" in text:
        return "Processing failed. The PDF could not be read by the existing pipeline."
    return text[:500]
