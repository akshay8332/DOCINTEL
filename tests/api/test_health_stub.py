"""Minimal API smoke (optional — skips if fastapi TestClient unavailable)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")


def test_health():
    from fastapi.testclient import TestClient

    from docintel.api.main import create_app

    client = TestClient(create_app())
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
