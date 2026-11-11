"""FastAPI app stub — Stage 2 expands routes and wiring."""

from __future__ import annotations

from fastapi import FastAPI

from docintel.api.routes_extract import router as extract_router
from docintel.api.routes_jobs import router as jobs_router


def create_app() -> FastAPI:
    app = FastAPI(title="DocIntel Extraction API", version="0.1.0")
    app.include_router(jobs_router)
    app.include_router(extract_router)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
