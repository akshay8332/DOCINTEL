"""Ingest service stub — accept uploads / storage URIs."""

from __future__ import annotations

from pathlib import Path

from docintel.schemas.jobs import JobCreate, JobStatus, JobState


def create_job(payload: JobCreate, *, storage_root: Path) -> JobStatus:
    storage_root.mkdir(parents=True, exist_ok=True)
    job_id = f"job-{len(list(storage_root.glob('*'))) + 1:05d}"
    return JobStatus(job_id=job_id, state=JobState.pending, document_id=None)
