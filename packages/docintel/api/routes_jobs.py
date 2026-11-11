"""Job routes stub (Stage 2)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from docintel.schemas.jobs import JobCreate, JobStatus
from docintel.services import ingest, store
from docintel.schemas.config import AppSettings

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


@router.post("", response_model=JobStatus)
def create_job(payload: JobCreate) -> JobStatus:
    settings = AppSettings()
    job = ingest.create_job(payload, storage_root=settings.storage_root)
    return store.save_job(job)


@router.get("/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job
