"""SQLite tables for uploaded invoices. Original predictions are never updated."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Invoice(Base):
    __tablename__ = "invoices"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    stored_name: Mapped[str] = mapped_column(String(64))
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    processing_status: Mapped[str] = mapped_column(String(32), default="uploaded")
    review_status: Mapped[str] = mapped_column(String(32), default="pending")
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    runs: Mapped[list[ProcessingRun]] = relationship(back_populates="invoice")


class ProcessingRun(Base):
    __tablename__ = "processing_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    invoice_id: Mapped[str] = mapped_column(ForeignKey("invoices.id"))
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    checkpoint_reference: Mapped[str | None] = mapped_column(String(255), nullable=True)
    correction_version: Mapped[str] = mapped_column(String(128))
    configuration: Mapped[dict] = mapped_column(JSON)
    processing_duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    mode: Mapped[str] = mapped_column(String(32), default="unavailable")

    invoice: Mapped[Invoice] = relationship(back_populates="runs")
    fields: Mapped[list[ExtractionResult]] = relationship(back_populates="run")


class ExtractionResult(Base):
    __tablename__ = "extraction_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    processing_run_id: Mapped[str] = mapped_column(ForeignKey("processing_runs.id"))
    field_name: Mapped[str] = mapped_column(String(64))
    original_text: Mapped[str] = mapped_column(Text, default="")
    corrected_text: Mapped[str] = mapped_column(Text, default="")
    approved_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bounding_box: Mapped[list | None] = mapped_column(JSON, nullable=True)
    box_0_1000: Mapped[list | None] = mapped_column(JSON, nullable=True)
    available_confidence_signals: Mapped[dict] = mapped_column(JSON)
    token_changes: Mapped[list] = mapped_column(JSON)
    layers_applied: Mapped[list] = mapped_column(JSON)
    review_status: Mapped[str] = mapped_column(String(32), default="pending")
    demo: Mapped[int] = mapped_column(Integer, default=0)

    run: Mapped[ProcessingRun] = relationship(back_populates="fields")
    reviews: Mapped[list[ReviewHistory]] = relationship(back_populates="result")


class ReviewHistory(Base):
    __tablename__ = "review_history"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    extraction_result_id: Mapped[str] = mapped_column(ForeignKey("extraction_results.id"))
    original_prediction: Mapped[str] = mapped_column(Text, default="")
    corrected_prediction: Mapped[str] = mapped_column(Text, default="")
    approved_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewer_action: Mapped[str] = mapped_column(String(32))
    reviewer_comment: Mapped[str] = mapped_column(Text, default="")
    correction_reason: Mapped[str] = mapped_column(Text, default="")
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    correction_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    result: Mapped[ExtractionResult] = relationship(back_populates="reviews")
