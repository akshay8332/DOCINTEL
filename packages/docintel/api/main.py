"""FastAPI application. The /health and /v1 stubs stay in place."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from docintel.api.deps import get_settings
from docintel.api.routes_extract import router as extract_router
from docintel.api.routes_invoices import router as invoice_router
from docintel.api.routes_jobs import router as jobs_router
from docintel.database.session import make_session_factory
from docintel.schemas.config import AppSettings
from docintel.services.address_pipeline import AddressPipeline
from docintel.services.invoice_service import InvoiceService


def create_app(settings: AppSettings | None = None) -> FastAPI:
    active = settings or get_settings()
    if active.invoice_db_url.startswith("sqlite:///./"):
        from pathlib import Path

        Path(active.invoice_db_url.removeprefix("sqlite:///./")).parent.mkdir(parents=True, exist_ok=True)
    app = FastAPI(
        title="DOCILE – Intelligent Invoice Processing",
        version="0.2.0",
        description=(
            "Local invoice address extraction. Real mode uses the frozen multi-label "
            "checkpoint and the existing correction stack. Demo mode is labeled and is "
            "not a model prediction."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    pipeline = AddressPipeline(active)
    app.state.settings = active
    app.state.pipeline = pipeline
    app.state.invoice_service = InvoiceService(make_session_factory(active.invoice_db_url), active, pipeline)
    app.include_router(jobs_router)
    app.include_router(extract_router)
    app.include_router(invoice_router)

    @app.get("/health")
    def health() -> dict:
        payload = pipeline.status()
        payload["status"] = "ok"
        return payload

    return app


app = create_app()
