"""Invoice upload, processing, and review."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from docintel.services.invoice_service import public_field, public_invoice, public_run

router = APIRouter(prefix="/api", tags=["invoices"])
_UPLOADS = File(...)


class ReviewBody(BaseModel):
    field_name: str
    approved_value: str = ""
    reviewer_comment: str = ""
    correction_reason: str = ""


class DecisionBody(BaseModel):
    reviewer_comment: str = ""


class ReferenceBody(BaseModel):
    vendor_address: str = ""
    customer_billing_address: str = ""
    note: str = Field(
        default="Supply reference text only when you have it. Predictions are not ground truth.",
    )


def _service(request: Request):
    return request.app.state.invoice_service


@router.get("/health")
def api_health(request: Request) -> dict:
    return request.app.state.pipeline.status()


@router.get("/invoices")
def list_invoices(request: Request) -> dict:
    service = _service(request)
    db = service.session()
    try:
        rows = service.list_invoices(db)
        return {"summary": service.summary(db), "invoices": [public_invoice(row) for row in rows]}
    finally:
        db.close()


@router.post("/invoices/upload")
async def upload_invoices(request: Request, files: list[UploadFile] = _UPLOADS) -> dict:
    service = _service(request)
    accepted = []
    rejected = []
    db = service.session()
    try:
        for item in files:
            content = await item.read()
            try:
                row = service.accept_upload(db, item.filename or "invoice.pdf", content)
            except ValueError as exc:
                rejected.append({"filename": item.filename or "invoice.pdf", "reason": str(exc)})
                continue
            accepted.append(public_invoice(row))
    finally:
        db.close()
    if not accepted:
        raise HTTPException(status_code=400, detail={"accepted": [], "rejected": rejected})
    return {"accepted": accepted, "rejected": rejected}


@router.get("/invoices/{invoice_id}")
def get_invoice(invoice_id: str, request: Request) -> dict:
    service = _service(request)
    db = service.session()
    try:
        row = service.get_invoice(db, invoice_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Invoice not found.")
        run = service.latest_run(db, invoice_id)
        payload = public_invoice(row)
        payload["latest_run"] = public_run(run) if run else None
        return payload
    finally:
        db.close()


@router.get("/invoices/{invoice_id}/file")
def invoice_file(invoice_id: str, request: Request):
    service = _service(request)
    db = service.session()
    try:
        row = service.get_invoice(db, invoice_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Invoice not found.")
        try:
            path = service.stored_file(row)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="The uploaded PDF is no longer available.") from None
        return FileResponse(path, media_type="application/pdf", filename=row.filename)
    finally:
        db.close()


@router.post("/invoices/{invoice_id}/process")
def process_invoice(invoice_id: str, request: Request) -> dict:
    service = _service(request)
    db = service.session()
    try:
        try:
            row = service.mark_processing(db, invoice_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Invoice not found.") from None
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        invoice = public_invoice(row)
    finally:
        db.close()
    service.start_processing(invoice_id)
    if service.settings.process_inline:
        db = service.session()
        try:
            return public_invoice(service.get_invoice(db, invoice_id))
        finally:
            db.close()
    return invoice


@router.get("/invoices/{invoice_id}/results")
def invoice_results(invoice_id: str, request: Request) -> dict:
    service = _service(request)
    db = service.session()
    try:
        row = service.get_invoice(db, invoice_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Invoice not found.")
        run = service.latest_run(db, invoice_id)
        return {
            "invoice": public_invoice(row),
            "run": public_run(run) if run else None,
            "fields": [public_field(item) for item in run.fields] if run else [],
        }
    finally:
        db.close()


@router.put("/invoices/{invoice_id}/review")
def review_invoice(invoice_id: str, body: ReviewBody, request: Request) -> dict:
    service = _service(request)
    db = service.session()
    try:
        try:
            result = service.save_edit(
                db,
                invoice_id,
                body.field_name,
                body.approved_value,
                body.reviewer_comment,
                body.correction_reason,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Invoice not found.") from None
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return public_field(result)
    finally:
        db.close()


@router.post("/invoices/{invoice_id}/approve")
def approve_invoice(invoice_id: str, body: DecisionBody, request: Request) -> dict:
    return _decide(invoice_id, "approved", body, request)


@router.post("/invoices/{invoice_id}/reject")
def reject_invoice(invoice_id: str, body: DecisionBody, request: Request) -> dict:
    return _decide(invoice_id, "rejected", body, request)


@router.get("/invoices/{invoice_id}/history")
def invoice_history(invoice_id: str, request: Request) -> dict:
    service = _service(request)
    db = service.session()
    try:
        try:
            return service.history(db, invoice_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Invoice not found.") from None
    finally:
        db.close()


@router.get("/invoices/{invoice_id}/export")
def export_invoice(invoice_id: str, request: Request, format: str = Query(default="json")):
    service = _service(request)
    db = service.session()
    try:
        row = service.get_invoice(db, invoice_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Invoice not found.")
        if format == "csv":
            try:
                content = service.export_csv(db, invoice_id)
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from None
            return Response(
                content,
                media_type="text/csv",
                headers={"Content-Disposition": f'attachment; filename="{row.filename}.csv"'},
            )
        payload = service.history(db, invoice_id)
        payload["invoice"] = public_invoice(row)
        return payload
    finally:
        db.close()


@router.post("/invoices/{invoice_id}/compare")
def compare_invoice(invoice_id: str, body: ReferenceBody, request: Request) -> dict:
    service = _service(request)
    db = service.session()
    try:
        if service.get_invoice(db, invoice_id) is None:
            raise HTTPException(status_code=404, detail="Invoice not found.")
        try:
            return service.compare_reference(
                db,
                invoice_id,
                {
                    "vendor_address": body.vendor_address,
                    "customer_billing_address": body.customer_billing_address,
                },
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
    finally:
        db.close()


def _decide(invoice_id: str, action: str, body: DecisionBody, request: Request) -> dict:
    service = _service(request)
    db = service.session()
    try:
        try:
            row = service.decide(db, invoice_id, action, body.reviewer_comment)
        except KeyError:
            raise HTTPException(status_code=404, detail="Invoice not found.") from None
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return public_invoice(row)
    finally:
        db.close()
