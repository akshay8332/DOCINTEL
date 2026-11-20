"""Fine-tune the frozen epoch-3 checkpoint on the focused party documents.

This entry point does not call the production trainer. The production trainer
always starts from microsoft/layoutlmv3-base and a new classification head.
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "experiments"))

from multilabel import FIELD_NAMES
from multilabel.checkpoint import (
    assert_output_allowed,
    build_scheduler,
    plan_resume,
    save_training_checkpoint,
)
from multilabel.data import collate_pages
from multilabel.device import describe_device
from multilabel.model import MultiLabelLayoutLM
from multilabel.train import (
    PageFiles,
    _optimizer,
    _run_epoch,
    _seed,
    _trainer_state,
    encode_split,
    encode_to_disk,
)

CONFIG = ROOT / "configs" / "kaggle_party_experiment.yaml"
FOCUSED_IDS = ROOT / "configs" / "party_experiment_focused_ids.json"
PRODUCTION_EPOCH3 = ROOT / "artifacts" / "checkpoints" / "epoch-3"


def load_experiment_settings(path: Path | None = None) -> dict:
    """Read the Kaggle recipe. Environment variables replace the mounted paths."""
    import yaml

    config_path = path or CONFIG
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings.setdefault("logging_steps", 10)
    settings.setdefault("gradient_accumulation_steps", 1)
    settings.setdefault("max_grad_norm", 1.0)
    settings.setdefault("spatial_gap", 30)
    settings.setdefault("patience", 2)
    settings.setdefault("min_delta", 0.0)
    settings.setdefault("seed", 42)
    settings.setdefault("batch_size", 2)
    settings.setdefault("epochs", 3)
    settings.setdefault("learning_rate", 5.0e-6)
    settings.setdefault("weight_decay", 0.01)
    settings.setdefault("warmup_ratio", 0.0)
    settings.setdefault("eval_limit", 20)
    settings.setdefault("max_seq_length", 512)
    settings.setdefault("threshold", 0.5)
    settings.setdefault("grouping", "line-block-v2")
    settings.setdefault("resume", False)
    settings.setdefault("require_gpu", True)
    settings.setdefault("model_name", "microsoft/layoutlmv3-base")
    settings.setdefault("run_name", "party-address-finetune")
    if os.environ.get("DOCINTEL_RESUME") == "1":
        settings["resume"] = True
    return settings


def focused_document_ids(path: Path | None = None) -> list[str]:
    payload = json.loads((path or FOCUSED_IDS).read_text(encoding="utf-8"))
    ids = payload["document_ids"] if isinstance(payload, dict) else payload
    if not ids:
        raise RuntimeError(f"No training documents in {path or FOCUSED_IDS}")
    return [str(doc_id) for doc_id in ids]


def assert_experiment_output(init_dir: Path, output_dir: Path) -> None:
    """Refuse to write the production epoch-3 checkpoint or the init directory."""
    init_resolved = init_dir.resolve()
    output_resolved = output_dir.resolve()
    production = PRODUCTION_EPOCH3.resolve()
    if output_resolved == production or production in output_resolved.parents:
        raise RuntimeError(f"Refusing to write the frozen checkpoint {production}")
    if output_resolved == init_resolved or init_resolved in output_resolved.parents:
        raise RuntimeError(f"Refusing to write the init checkpoint {init_resolved}")
    if output_resolved in init_resolved.parents:
        raise RuntimeError(f"Refusing to write above the init checkpoint {init_resolved}")
    assert_output_allowed(output_resolved)


def _search_roots() -> list[Path]:
    roots = []
    kaggle = Path("/kaggle/input")
    if kaggle.is_dir():
        roots.append(kaggle)
    return roots


def resolve_data_dir(settings: dict) -> Path:
    env = os.environ.get("DOCINTEL_DATA_DIR")
    if env:
        path = Path(env)
        if (path / "train.json").is_file():
            return path
        raise RuntimeError(f"DOCINTEL_DATA_DIR has no train.json: {path}")
    configured = Path(str(settings.get("data_dir", "")))
    if (configured / "train.json").is_file():
        return configured
    found = [path.parent for root in _search_roots() for path in root.rglob("train.json")]
    if len(found) == 1:
        return found[0]
    if not found:
        raise RuntimeError(
            "DocILE data was not found. Attach it on Kaggle and set DOCINTEL_DATA_DIR "
            "to the folder that contains train.json, ocr, annotations, and pdfs."
        )
    listed = "\n".join(str(path) for path in found)
    raise RuntimeError(f"Several DocILE folders were found. Set DOCINTEL_DATA_DIR to one:\n{listed}")


def resolve_init_checkpoint(settings: dict) -> Path:
    env = os.environ.get("DOCINTEL_INIT_CHECKPOINT")
    candidates = []
    if env:
        candidates.append(Path(env))
    if settings.get("init_checkpoint"):
        configured = Path(str(settings["init_checkpoint"]))
        if not configured.is_absolute():
            configured = ROOT / configured
        candidates.append(configured)
    candidates.append(PRODUCTION_EPOCH3)
    for path in candidates:
        if (path / "model.pt").is_file():
            return path
    found = [
        path.parent
        for root in _search_roots()
        for path in root.rglob("model.pt")
        if (path.parent / "tokenizer_config.json").is_file()
    ]
    if len(found) == 1:
        return found[0]
    if not found:
        raise RuntimeError(
            "The epoch-3 checkpoint was not found. Add that folder as a Kaggle dataset "
            "and set DOCINTEL_INIT_CHECKPOINT to the folder that contains model.pt."
        )
    listed = "\n".join(str(path) for path in found)
    raise RuntimeError(f"Several checkpoints were found. Set DOCINTEL_INIT_CHECKPOINT to one:\n{listed}")


def resolve_output_dir(settings: dict) -> Path:
    env = os.environ.get("DOCINTEL_OUTPUT_DIR")
    if env:
        return Path(env)
    if Path("/kaggle/working").is_dir():
        return Path("/kaggle/working/docile_party_experiment")
    configured = Path(str(settings.get("output_dir", "artifacts/checkpoints/docile_party_experiment")))
    if not configured.is_absolute():
        configured = ROOT / configured
    return configured


def load_epoch3_weights(model: MultiLabelLayoutLM, init_dir: Path, device: str) -> None:
    """Load encoder and classifier tensors. Adam moments and the old RNG stay unused."""
    payload = torch.load(init_dir / "model.pt", map_location=device, weights_only=False)
    names = list(payload.get("field_names") or [])
    if names != list(FIELD_NAMES):
        raise RuntimeError(
            "The init checkpoint field order does not match the nine production labels. "
            f"Checkpoint labels: {names}"
        )
    model.encoder.load_state_dict(payload["encoder"])
    model.classifier.load_state_dict(payload["classifier"])
    del payload


def _processor(init_dir: Path, model_name: str):
    from transformers import LayoutLMv3Processor

    if (init_dir / "tokenizer_config.json").is_file():
        return LayoutLMv3Processor.from_pretrained(init_dir, apply_ocr=False)
    return LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    config_path = CONFIG
    if args and not args[0].startswith("--"):
        config_path = Path(args[0])
    settings = load_experiment_settings(config_path)
    init_dir = resolve_init_checkpoint(settings)
    output_dir = resolve_output_dir(settings)
    data_dir = resolve_data_dir(settings)
    document_ids = focused_document_ids()
    assert_experiment_output(init_dir, output_dir)
    print(f"Init checkpoint (read only): {init_dir}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"DocILE: {data_dir}", flush=True)
    print(f"Focused training documents: {len(document_ids)}", flush=True)
    if "--dry-run" in args:
        print("Dry run complete. Training was not started.", flush=True)
        return 0

    try:
        device = describe_device(bool(settings.get("require_gpu", False)))
    except RuntimeError as exc:
        print(exc, flush=True)
        return 1
    _seed(int(settings["seed"]), device)
    init_stamp = (init_dir / "model.pt").stat().st_mtime_ns
    from transformers import LayoutLMv3Model

    processor = _processor(init_dir, settings["model_name"])
    image_dir = output_dir / "page_images"
    from docintel.data.docile import iter_docile_pages

    print("Encoding validation pages", flush=True)
    val_docs = list(
        iter_docile_pages(
            data_dir,
            "val",
            limit=int(settings["eval_limit"]),
            image_dir=image_dir,
        )
    )
    print(f"Validation pages={len(val_docs)}", flush=True)
    max_length = int(settings["max_seq_length"])
    print("Encoding focused training pages", flush=True)
    train_paths, train_audit = encode_to_disk(
        iter_docile_pages(
            data_dir,
            "train",
            image_dir=image_dir,
            document_ids=set(document_ids),
        ),
        processor,
        max_length,
        output_dir / "encoded",
    )
    train_pages = PageFiles(train_paths)
    print(f"Train pages={len(train_pages)}", flush=True)
    val_pages = encode_split(val_docs, processor, max_length)
    del val_docs
    gc.collect()
    if not train_pages:
        print("No focused training pages were encoded. Training stopped.", flush=True)
        return 1

    encoder = LayoutLMv3Model.from_pretrained(settings["model_name"])
    model = MultiLabelLayoutLM(encoder, num_labels=len(FIELD_NAMES)).to(device)
    load_epoch3_weights(model, init_dir, device)
    print("Loaded epoch-3 weights. Optimizer moments start at the new learning rate.", flush=True)
    optimizer = _optimizer(model, float(settings["learning_rate"]), float(settings["weight_decay"]))
    train_loader = DataLoader(
        train_pages,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        collate_fn=collate_pages,
    )
    val_loader = DataLoader(
        val_pages,
        batch_size=int(settings["batch_size"]),
        shuffle=False,
        collate_fn=collate_pages,
    )
    accum = max(1, int(settings["gradient_accumulation_steps"]))
    updates_per_epoch = (len(train_loader) + accum - 1) // accum
    scheduler = build_scheduler(
        optimizer,
        settings,
        max(1, updates_per_epoch * int(settings["epochs"])),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        planned = plan_resume(
            output_dir,
            resume=bool(settings.get("resume", False)),
            min_delta=float(settings["min_delta"]),
            patience=int(settings["patience"]),
        )
    except RuntimeError as exc:
        print(exc, flush=True)
        return 1
    history = list(planned["history"])
    best_loss = planned["best_val_loss"]
    best_epoch = planned["best_epoch"]
    wait = planned["wait"]
    stopped_early = planned["stopped_early"]
    start_epoch = planned["start_epoch"]
    if planned["restored"]:
        from multilabel.checkpoint import load_training_checkpoint

        load_training_checkpoint(model, planned["epoch_dir"], optimizer, scheduler, device)
        print(f"Continuing from epoch {start_epoch - 1}", flush=True)
    if stopped_early:
        print("Early stopping already reached", flush=True)
        start_epoch = int(settings["epochs"]) + 1
    log_path = output_dir / "train_log.jsonl"
    min_delta = float(settings["min_delta"])
    for epoch in range(start_epoch, int(settings["epochs"]) + 1):
        train_loss = _run_epoch(
            model, train_loader, optimizer, device, settings, train=True, scheduler=scheduler
        )
        val_loss = _run_epoch(model, val_loader, optimizer, device, settings, train=False)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        best_updated = False
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_epoch = epoch
            wait = 0
            best_updated = True
        else:
            wait += 1
            if wait >= int(settings["patience"]):
                stopped_early = True
        state = _trainer_state(
            epoch, history, best_loss, best_epoch, wait, stopped_early, settings, train_pages
        )
        state["initialization"] = str(init_dir)
        state["focused_documents"] = len(document_ids)
        state["train_audit"] = train_audit
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        (output_dir / "trainer_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"Epoch {epoch} train {train_loss:.4f} val {val_loss:.4f}", flush=True)
        save_training_checkpoint(
            model,
            output_dir / f"epoch-{epoch}",
            field_names=list(FIELD_NAMES),
            model_name=settings["model_name"],
            optimizer=optimizer,
            scheduler=scheduler,
            processor=processor,
            trainer_state=state,
            extra=row,
        )
        if best_updated:
            save_training_checkpoint(
                model,
                output_dir / "best",
                field_names=list(FIELD_NAMES),
                model_name=settings["model_name"],
                optimizer=optimizer,
                scheduler=scheduler,
                processor=processor,
                trainer_state=state,
                extra=row,
            )
        if stopped_early:
            print("Early stopping", flush=True)
            break

    summary = {
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "best_val_loss": best_loss,
        "best_epoch": best_epoch,
        "history": history,
        "validation_used_for_gradients": False,
        "train_pages": len(train_pages),
        "focused_documents": len(document_ids),
        "initialization": str(init_dir),
        "learning_rate": float(settings["learning_rate"]),
        "baseline_checkpoint": "epoch-3",
        "baseline_raw_exact": "61/162",
        "baseline_corrected_exact": "72/162",
    }
    (output_dir / "train_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if (init_dir / "model.pt").stat().st_mtime_ns != init_stamp:
        raise RuntimeError(f"The init checkpoint changed during training: {init_dir}")
    print(f"Saved {output_dir}", flush=True)
    print(
        "Score this checkpoint locally on the labelled 100-page set. "
        "Adopt it only when raw exact is above 61/162, corrected exact is above 72/162, "
        "customer raw exact is at least 28/70, and false-positive words are not above 171.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
