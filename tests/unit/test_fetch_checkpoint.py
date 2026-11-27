"""Checkpoint fetch skips when S3 is not configured."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "scripts" / "fetch_checkpoint.py"


def _load():
    spec = importlib.util.spec_from_file_location("fetch_checkpoint", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["fetch_checkpoint"] = module
    spec.loader.exec_module(module)
    return module


def test_fetch_checkpoint_skips_without_uri(monkeypatch):
    module = _load()
    monkeypatch.delenv("DOCINTEL_S3_CHECKPOINT_URI", raising=False)
    assert module.main() == 0


def test_fetch_checkpoint_rejects_non_s3_uri(monkeypatch):
    module = _load()
    monkeypatch.setenv("DOCINTEL_S3_CHECKPOINT_URI", "https://example.com/model")
    assert module.main() == 1
