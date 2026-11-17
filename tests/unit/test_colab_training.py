"""GPU selection, checkpoint resume, frozen baselines, and DocILE export."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "packages"))

from multilabel import FIELD_NAMES
from multilabel.checkpoint import (
    assert_diagnostic_destination,
    assert_output_allowed,
    build_scheduler,
    capture_rng,
    latest_complete_epoch,
    load_training_checkpoint,
    plan_resume,
    resolve_run_dirs,
    restore_latest_from_persistent,
    restore_rng,
    save_training_checkpoint,
    sync_checkpoint,
)
from multilabel.data import load_settings
from multilabel.device import describe_device
from multilabel.evaluate import extraction_scorecard
from multilabel.group import group_fields
from multilabel.model import MultiLabelLayoutLM


class _TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 4})()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, **kwargs):
        batch, length = kwargs["input_ids"].shape
        hidden = self.proj.weight.new_zeros(batch, length, 4)
        hidden = self.proj(hidden)
        return type("Out", (), {"last_hidden_state": hidden})()


def _fitted_pair():
    model = MultiLabelLayoutLM(_TinyEncoder(), num_labels=9)
    optimizer = AdamW(model.parameters(), lr=0.02)
    scheduler = build_scheduler(optimizer, {"warmup_ratio": 0.0}, total_steps=8)
    logits, loss = model(
        input_ids=torch.zeros(1, 2, dtype=torch.long),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        bbox=torch.zeros(1, 2, 4, dtype=torch.long),
        pixel_values=torch.zeros(1, 3, 8, 8),
        labels=torch.ones(1, 2, 9),
        loss_mask=torch.ones(1, 2),
    )
    assert logits.shape[-1] == 9
    loss.backward()
    optimizer.step()
    scheduler.step()
    return model, optimizer, scheduler


def test_checkpoint_saves_and_restores_weights_optimizer_and_scheduler(tmp_path):
    model, optimizer, scheduler = _fitted_pair()
    weight = model.classifier.weight.detach().clone()
    moment = optimizer.state[model.classifier.weight]["exp_avg"].detach().clone()
    step = scheduler.last_epoch
    state = {
        "epoch": 1,
        "best_val_loss": 0.2,
        "best_epoch": 1,
        "wait": 0,
        "stopped_early": False,
        "history": [{"epoch": 1, "train_loss": 0.4, "val_loss": 0.2}],
    }
    save_training_checkpoint(
        model,
        tmp_path / "epoch-1",
        field_names=list(FIELD_NAMES),
        model_name="tiny",
        optimizer=optimizer,
        scheduler=scheduler,
        trainer_state=state,
    )
    model.classifier.weight.data.add_(4)
    restored, restored_optimizer, restored_scheduler = _fitted_pair()
    load_training_checkpoint(
        restored,
        tmp_path / "epoch-1",
        restored_optimizer,
        restored_scheduler,
    )
    assert torch.allclose(restored.classifier.weight, weight)
    assert torch.allclose(restored_optimizer.state[restored.classifier.weight]["exp_avg"], moment)
    assert restored_scheduler.last_epoch == step
    assert (tmp_path / "epoch-1" / "complete.json").is_file()
    assert not (tmp_path / "epoch-1" / "writing.json").exists()


def test_rng_state_restores_the_next_draw():
    torch.manual_seed(123)
    state = capture_rng()
    expected = torch.rand(1).item()
    torch.manual_seed(999)
    restore_rng(state)
    assert torch.rand(1).item() == expected


def test_resume_continues_from_the_saved_epoch_and_refuses_a_restart(tmp_path):
    model, optimizer, scheduler = _fitted_pair()
    save_training_checkpoint(
        model,
        tmp_path / "epoch-1",
        field_names=list(FIELD_NAMES),
        model_name="tiny",
        optimizer=optimizer,
        scheduler=scheduler,
        trainer_state={
            "epoch": 1,
            "best_val_loss": 0.037,
            "best_epoch": 1,
            "wait": 0,
            "stopped_early": False,
            "history": [{"epoch": 1, "train_loss": 0.1, "val_loss": 0.037}],
        },
    )
    planned = plan_resume(tmp_path, resume=True, patience=2)
    assert planned["restored"] is True
    assert planned["start_epoch"] == 2
    assert planned["best_val_loss"] == pytest.approx(0.037)
    with pytest.raises(RuntimeError, match="not restarted"):
        plan_resume(tmp_path, resume=False)


def test_a_half_written_epoch_is_not_a_resume_point(tmp_path):
    epoch = tmp_path / "epoch-1"
    epoch.mkdir()
    (epoch / "model.pt").write_bytes(b"partial")
    (epoch / "writing.json").write_text("{}\n", encoding="utf-8")
    assert latest_complete_epoch(tmp_path) is None
    assert plan_resume(tmp_path, resume=True)["start_epoch"] == 1


def test_drive_sync_keeps_earlier_epochs_and_skips_encoded_pages(tmp_path):
    work = tmp_path / "work"
    drive = tmp_path / "drive"
    previous = drive / "epoch-1"
    previous.mkdir(parents=True)
    (previous / "keep.txt").write_text("safe", encoding="utf-8")
    current = work / "epoch-2"
    current.mkdir(parents=True)
    (current / "model.pt").write_bytes(b"new")
    (work / "encoded").mkdir(parents=True)
    (work / "encoded" / "00001.pt").write_bytes(b"page")
    (work / "train_log.jsonl").write_text('{"epoch": 2}\n', encoding="utf-8")
    sync_checkpoint(work, drive, 2, best_updated=False)
    assert (drive / "epoch-1" / "keep.txt").read_text(encoding="utf-8") == "safe"
    assert (drive / "epoch-2" / "model.pt").read_bytes() == b"new"
    assert not (drive / "encoded").exists()
    assert (drive / "train_log.jsonl").is_file()


def test_persistent_restore_copies_the_latest_epoch_only(tmp_path):
    drive = tmp_path / "drive"
    work = tmp_path / "work"
    for epoch, payload in ((1, b"one"), (2, b"two")):
        directory = drive / f"epoch-{epoch}"
        directory.mkdir(parents=True)
        (directory / "model.pt").write_bytes(payload)
        (directory / "complete.json").write_text("{}\n", encoding="utf-8")
    restore_latest_from_persistent(work, drive)
    assert (work / "epoch-2" / "model.pt").read_bytes() == b"two"
    assert not (work / "epoch-1").exists()
    assert (drive / "epoch-1" / "model.pt").read_bytes() == b"one"


def test_frozen_checkpoints_and_diagnostics_are_rejected(tmp_path):
    assert_output_allowed(tmp_path / "artifacts" / "checkpoints" / "docile_multilabel_full_5180")
    for name in (
        "invoice-layoutlm-200",
        "docile_multilabel_200",
        "docile_multilabel_200_baseline",
        "docile_multilabel_full_5180_epoch3",
    ):
        with pytest.raises(RuntimeError, match="frozen"):
            assert_output_allowed(tmp_path / "artifacts" / "checkpoints" / name / "best")
    with pytest.raises(RuntimeError, match="frozen diagnostic"):
        assert_diagnostic_destination(Path("multilabel-reconstruction-100.json"))
    assert_diagnostic_destination(Path("multilabel-full-5180-vs-epoch8.json"))


def test_cpu_is_reported_and_a_required_gpu_stops_training(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert describe_device(False) == "cpu"
    printed = capsys.readouterr().out
    assert "PyTorch" in printed
    assert "CUDA version" in printed
    with pytest.raises(RuntimeError, match="Training stopped"):
        describe_device(True)


def test_cuda_is_selected_when_it_is_available(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "Test GPU")
    monkeypatch.setattr(torch.version, "cuda", "12.4", raising=False)
    assert describe_device(True) == "cuda"
    printed = capsys.readouterr().out
    assert "Test GPU" in printed
    assert "12.4" in printed
    assert torch.__version__ in printed


def test_zero_warmup_leaves_the_learning_rate_unchanged():
    _model, optimizer, scheduler = _fitted_pair()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.02)


def test_colab_scratch_paths_are_ignored_until_that_disk_exists(tmp_path):
    settings = {
        "output_dir": str(tmp_path / "artifacts" / "checkpoints" / "docile_multilabel_full_5180"),
        "scratch_dir": "/content/checkpoints/docile_multilabel_full_5180",
        "persistent_dir": "/content/drive/MyDrive/docintel/artifacts/checkpoints/docile_multilabel_full_5180",
        "require_persistent": False,
    }
    work, persistent = resolve_run_dirs(settings)
    if not Path("/content").exists():
        assert work == Path(settings["output_dir"])
        assert persistent is None
    settings["require_persistent"] = True
    if not Path("/content/drive/MyDrive").exists():
        with pytest.raises(RuntimeError, match="Training stopped"):
            resolve_run_dirs(settings)


def test_settings_keep_colab_paths_literal(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "colab.yaml").write_text(
        "\n".join(
            [
                "data_dir: /content/docile",
                "output_dir: ./artifacts/checkpoints/docile_multilabel_full_5180",
                "scratch_dir: /content/checkpoints/docile_multilabel_full_5180",
                "persistent_dir: /content/drive/MyDrive/docintel/artifacts/checkpoints/docile_multilabel_full_5180",
                "docile_src: /content/docile-main",
                "softmax_report: ./artifacts/diagnostics/phase2-report.json",
                "weights_checkpoint: ./artifacts/checkpoints/docile_multilabel_200_baseline",
            ]
        ),
        encoding="utf-8",
    )
    settings = load_settings(config_dir / "colab.yaml")
    assert settings["data_dir"] == "/content/docile"
    assert settings["scratch_dir"].startswith("/content/")
    assert settings["output_dir"].endswith("docile_multilabel_full_5180")


def test_prediction_export_uses_docile_fields():
    fields = group_fields(
        ["INV-1"],
        [[0, 0, 100, 20]],
        [[0.8, 0, 0, 0, 0, 0, 0, 0, 0]],
        list(FIELD_NAMES),
        threshold=0.5,
        page_index=1,
    )
    assert len(fields) == 1
    payload = fields[0].to_docile_dict()
    assert payload["fieldtype"] == "document_id"
    assert payload["text"] == "INV-1"
    assert payload["page"] == 1
    assert payload["score"] == pytest.approx(0.8)
    assert payload["use_only_for_ap"] is False
    assert payload["line_item_id"] is None
    assert all(0.0 <= value <= 1.0 for value in payload["bbox"])


def test_scorecard_reports_ap_f1_and_field_scores():
    official = {
        "iou_1.0": {
            "all_kile_fields": {"AP": 0.3},
            "nine_trained_fields": {"f1": 0.5, "precision": 0.6, "recall": 0.4},
            "per_field": {"document_id": {"precision": 1, "recall": 1, "f1": 1, "AP": 1}},
        },
        "iou_0.5": {
            "all_kile_fields": {"AP": 0.4},
            "nine_trained_fields": {"f1": 0.641, "precision": 0.7, "recall": 0.6},
            "per_field": {},
        },
        "iou_1.0_exact_text": {
            "all_kile_fields": {"AP": 0.1},
            "nine_trained_fields": {"f1": 0.42, "precision": 0.4, "recall": 0.4},
            "per_field": {},
        },
    }
    card = extraction_scorecard(official)
    assert card["ap_overlap_1.0"] == 0.3
    assert card["f1_overlap_1.0"] == 0.5
    assert card["f1_overlap_0.5"] == 0.641
    assert card["f1_exact_text"] == 0.42
    assert card["overlap_1.0"]["per_field"]["document_id"]["f1"] == 1
