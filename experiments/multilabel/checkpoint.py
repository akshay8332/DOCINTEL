"""Epoch checkpoints, resume, and a once-per-epoch copy to durable storage."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import torch

FROZEN_CHECKPOINT_DIRS = (
    "invoice-layoutlm-200",
    "docile_multilabel_200",
    "docile_multilabel_200_baseline",
)

FROZEN_DIAGNOSTICS = frozenset(
    {
        "multilabel-comparison.json",
        "multilabel-val-predictions.json",
        "multilabel-eval.json",
        "multilabel-reconstruction.json",
        "multilabel-reconstruction-100.json",
        "multilabel-targets.json",
        "phase4-diagnostics.json",
        "role-line-audit.json",
        "party-boundary-annotations.json",
        "inference-baselines.json",
        "phase2-report.json",
        "invoice-200.json",
        "multilabel-word-probe.json",
        "multilabel-word-scores-100.jsonl",
    }
)

_RUN_FILES = ("train_log.jsonl", "trainer_state.json", "train_metrics.json")


def assert_output_allowed(output_dir: Path) -> None:
    """Refuse any write that lands inside a frozen checkpoint directory."""
    resolved = output_dir.resolve()
    frozen = [part for part in resolved.parts if part in FROZEN_CHECKPOINT_DIRS]
    if frozen:
        raise RuntimeError(f"Refusing to write the frozen checkpoint directory {frozen[0]}")


def assert_diagnostic_destination(path: Path) -> None:
    """Refuse the published baseline reports. New runs use a different file name."""
    if path.name in FROZEN_DIAGNOSTICS:
        raise RuntimeError(f"Refusing to overwrite frozen diagnostic results in {path.name}")


def build_scheduler(optimizer, settings: dict, total_steps: int):
    """Constant learning rate when ``warmup_ratio`` is 0, matching the 200-page run.

    The scheduler is still saved so a resumed run restores the same step count.
    ``warmup_ratio`` 0 leaves every update at the configured learning rate.
    """
    warmup_steps = int(total_steps * float(settings.get("warmup_ratio", 0.0)))

    def lr_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def capture_rng() -> dict:
    state = {"torch": torch.get_rng_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:
        pass
    return state


def restore_rng(state: dict) -> None:
    torch_state = state["torch"]
    if torch.is_tensor(torch_state):
        torch_state = torch_state.cpu()
    torch.set_rng_state(torch_state)
    random.setstate(state["python"])
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])


def save_training_checkpoint(
    model,
    directory: Path,
    *,
    field_names: list[str],
    model_name: str,
    optimizer=None,
    scheduler=None,
    processor=None,
    trainer_state: dict | None = None,
    extra: dict | None = None,
) -> None:
    """Write one finished epoch. ``complete.json`` is written last."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "writing.json").write_text('{"status": "writing"}\n', encoding="utf-8")
    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "classifier": model.classifier.state_dict(),
            "field_names": list(field_names),
            "model_name": model_name,
            **(extra or {}),
        },
        directory / "model.pt",
    )
    if optimizer is not None:
        torch.save(optimizer.state_dict(), directory / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), directory / "scheduler.pt")
    torch.save(capture_rng(), directory / "rng.pt")
    if trainer_state is not None:
        (directory / "trainer_state.json").write_text(
            json.dumps(trainer_state, indent=2),
            encoding="utf-8",
        )
    if processor is not None:
        processor.save_pretrained(directory)
    epoch = None if trainer_state is None else trainer_state.get("epoch")
    (directory / "complete.json").write_text(
        json.dumps({"epoch": epoch, "complete": True}),
        encoding="utf-8",
    )
    writing = directory / "writing.json"
    if writing.is_file():
        writing.unlink()


def epoch_is_complete(epoch_dir: Path) -> bool:
    model = epoch_dir / "model.pt"
    if not model.is_file() or model.stat().st_size <= 0:
        return False
    if (epoch_dir / "complete.json").is_file():
        return True
    # A checkpoint that stopped while its files were being written is not usable.
    if (epoch_dir / "writing.json").is_file():
        return False
    # Checkpoints written before complete.json existed are still usable.
    return True


def latest_complete_epoch(directory: Path) -> int | None:
    if not directory.is_dir():
        return None
    found: list[int] = []
    for child in directory.glob("epoch-*"):
        if not child.is_dir():
            continue
        suffix = child.name.split("-", 1)[1]
        if suffix.isdigit() and epoch_is_complete(child):
            found.append(int(suffix))
    return max(found) if found else None


def load_training_checkpoint(model, directory: Path, optimizer=None, scheduler=None, device: str = "cpu") -> dict:
    payload = torch.load(directory / "model.pt", map_location=device, weights_only=False)
    model.encoder.load_state_dict(payload["encoder"])
    model.classifier.load_state_dict(payload["classifier"])
    optimizer_path = directory / "optimizer.pt"
    if optimizer is not None and optimizer_path.is_file():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device, weights_only=False))
    elif optimizer is not None:
        print(f"No optimizer state in {directory.name}; Adam moments start again.", flush=True)
    scheduler_path = directory / "scheduler.pt"
    if scheduler is not None and scheduler_path.is_file():
        scheduler.load_state_dict(torch.load(scheduler_path, map_location=device, weights_only=False))
    rng_path = directory / "rng.pt"
    if rng_path.is_file():
        restore_rng(torch.load(rng_path, weights_only=False))
    elif "rng" in payload:
        torch_state = payload["rng"]
        torch.set_rng_state(torch_state.cpu() if torch.is_tensor(torch_state) else torch_state)
        if "py_rng" in payload:
            random.setstate(payload["py_rng"])
    state_path = directory / "trainer_state.json"
    if not state_path.is_file():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def plan_resume(work_dir: Path, resume: bool, min_delta: float = 0.0, patience: int = 2) -> dict:
    """Continue after the last finished epoch. Never start over when resume is requested."""
    latest = latest_complete_epoch(work_dir)
    empty = {
        "start_epoch": 1,
        "restored": False,
        "best_val_loss": float("inf"),
        "best_epoch": None,
        "wait": 0,
        "stopped_early": False,
        "history": [],
        "epoch_dir": None,
    }
    if latest is None:
        return empty
    if not resume:
        raise RuntimeError(
            f"Checkpoint epoch {latest} already exists in {work_dir}. "
            "Set resume: true to continue. Training was not restarted."
        )
    epoch_dir = work_dir / f"epoch-{latest}"
    state_path = epoch_dir / "trainer_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        history = list(state.get("history") or [])
        best_val_loss = float(state.get("best_val_loss", float("inf")))
        wait = int(state.get("wait", 0))
        best_epoch = state.get("best_epoch")
        stopped = bool(state.get("stopped_early", False)) or wait >= int(patience)
    else:
        history = _history_from_log(work_dir)
        best_val_loss = float("inf")
        wait = 0
        best_epoch = None
        for row in history:
            if float(row["val_loss"]) < best_val_loss - float(min_delta):
                best_val_loss = float(row["val_loss"])
                wait = 0
                best_epoch = row["epoch"]
            else:
                wait += 1
        stopped = wait >= int(patience)
    return {
        "start_epoch": int(latest) + 1,
        "restored": True,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "wait": wait,
        "stopped_early": stopped,
        "history": history,
        "epoch_dir": epoch_dir,
    }


def _history_from_log(work_dir: Path) -> list[dict]:
    log_path = work_dir / "train_log.jsonl"
    if not log_path.is_file():
        return []
    rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(json.loads(line))
    return rows


def _path_usable(value: str) -> bool:
    text = str(value).replace("\\", "/")
    if text.startswith("/content"):
        return Path("/content").exists()
    path = Path(value)
    if path.exists():
        return True
    parent = path.parent
    return parent.exists()


def resolve_run_dirs(settings: dict) -> tuple[Path, Path | None]:
    """Train on fast local disk and keep a separate durable copy when it exists."""
    output = Path(settings["output_dir"])
    work = output
    scratch = settings.get("scratch_dir")
    if scratch and _path_usable(str(scratch)):
        work = Path(scratch)
    persistent = None
    raw_persistent = settings.get("persistent_dir")
    if raw_persistent:
        if _path_usable(str(raw_persistent)):
            persistent = Path(raw_persistent)
        elif settings.get("require_persistent"):
            raise RuntimeError(
                "The persistent checkpoint directory is unavailable. "
                "Mount Google Drive before training. Training stopped."
            )
    return work, persistent


def _copy_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            _copy_tree(item, target)
        else:
            shutil.copy2(item, target)


def sync_checkpoint(work_dir: Path, persistent_dir: Path | None, epoch: int, *, best_updated: bool) -> None:
    """Copy one finished epoch to durable storage. Earlier epochs are left in place."""
    if persistent_dir is None:
        return
    source = work_dir / f"epoch-{epoch}"
    if source.is_dir():
        _copy_tree(source, persistent_dir / f"epoch-{epoch}")
    if best_updated and (work_dir / "best").is_dir():
        _copy_tree(work_dir / "best", persistent_dir / "best")
    for name in _RUN_FILES:
        path = work_dir / name
        if path.is_file():
            persistent_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, persistent_dir / name)
    evaluation = work_dir / "eval"
    if evaluation.is_dir():
        _copy_tree(evaluation, persistent_dir / "eval")


def restore_latest_from_persistent(work_dir: Path, persistent_dir: Path | None) -> None:
    """Bring the newest Drive epoch back onto local disk after a runtime restart."""
    if persistent_dir is None:
        return
    remote = latest_complete_epoch(persistent_dir)
    local = latest_complete_epoch(work_dir)
    if remote is None or (local is not None and local >= remote):
        return
    work_dir.mkdir(parents=True, exist_ok=True)
    _copy_tree(persistent_dir / f"epoch-{remote}", work_dir / f"epoch-{remote}")
    if (persistent_dir / "best").is_dir():
        _copy_tree(persistent_dir / "best", work_dir / "best")
    for name in _RUN_FILES:
        path = persistent_dir / name
        if path.is_file():
            shutil.copy2(path, work_dir / name)
    print(f"Restored epoch {remote} from {persistent_dir}", flush=True)
