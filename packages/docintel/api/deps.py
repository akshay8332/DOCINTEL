"""Shared FastAPI dependencies."""

from __future__ import annotations

from functools import lru_cache

from docintel.schemas.config import AppSettings


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()
