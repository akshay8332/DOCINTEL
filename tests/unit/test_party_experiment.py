"""The party experiment refuses to write epoch-3 and keeps the focused id list."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "packages"))

from multilabel.party_experiment import (
    FOCUSED_IDS,
    assert_experiment_output,
    focused_document_ids,
    load_experiment_settings,
)


def test_focused_ids_are_unique_training_ids():
    ids = focused_document_ids()
    assert len(ids) == 2519
    assert len(set(ids)) == 2519
    train = ROOT / "data" / "docile" / "train.json"
    if train.is_file():
        official = set(json.loads(train.read_text(encoding="utf-8")))
        assert set(ids) <= official


def test_settings_do_not_require_the_production_trainer_keys():
    settings = load_experiment_settings()
    assert settings["learning_rate"] == 5.0e-6
    assert settings["epochs"] == 3
    assert "docile_src" not in settings
    assert FOCUSED_IDS.is_file()


def test_output_cannot_replace_epoch3_or_the_init_checkpoint(tmp_path):
    init = tmp_path / "epoch-3"
    init.mkdir()
    allowed = tmp_path / "docile_party_experiment"
    assert_experiment_output(init, allowed)
    with pytest.raises(RuntimeError):
        assert_experiment_output(init, init)
    with pytest.raises(RuntimeError):
        assert_experiment_output(init, init / "epoch-1")
    production = ROOT / "artifacts" / "checkpoints" / "epoch-3"
    with pytest.raises(RuntimeError):
        assert_experiment_output(init, production)
