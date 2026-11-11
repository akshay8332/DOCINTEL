"""Job / field persistence stub (SQLite-first in Stage 2)."""

from __future__ import annotations

from docintel.schemas.jobs import JobStatus


_JOBS: dict[str, JobStatus] = {}


def save_job(job: JobStatus) -> JobStatus:
    _JOBS[job.job_id] = job
    return job


def get_job(job_id: str) -> JobStatus | None:
    return _JOBS.get(job_id)


def list_jobs() -> list[JobStatus]:
    return list(_JOBS.values())
