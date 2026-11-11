"""Optional queue consumer stub (Stage 2+)."""

from __future__ import annotations


def process_job(job_id: str) -> dict:
    """Placeholder: load job → preprocess → extract → validate → store."""
    raise NotImplementedError(
        f"process_job({job_id!r}) is a Stage 2+ stub — wire services.extract here."
    )


def main() -> None:
    print("process_job worker stub — not implemented yet.")


if __name__ == "__main__":
    main()
