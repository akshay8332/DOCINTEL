"""Train the multi-label head. Validation pages are scored, never used for updates."""

from __future__ import annotations

import gc
import json
import random
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "experiments"))

from multilabel import FIELD_NAMES
from multilabel.checkpoint import (
    assert_diagnostic_destination,
    build_scheduler,
    load_training_checkpoint,
    plan_resume,
    resolve_run_dirs,
    restore_latest_from_persistent,
    save_training_checkpoint,
    sync_checkpoint,
)
from multilabel.data import (
    assert_softmax_checkpoint_untouched,
    audit_encoded,
    collate_pages,
    encode_document,
    load_settings,
    prepare_documents,
    write_json,
)
from multilabel.device import describe_device
from multilabel.targets import pool_target_audit
from multilabel.evaluate import (
    comparison_report,
    extraction_scorecard,
    load_multilabel,
    official_kile,
    predict_pages,
)
from multilabel.model import MultiLabelLayoutLM


def _seed(seed: int, device: str) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def _optimizer(model, learning_rate: float, weight_decay: float) -> AdamW:
    decay, plain = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith(".bias"):
            plain.append(parameter)
        else:
            decay.append(parameter)
    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": plain, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def _move(batch: dict, device: str) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def _run_epoch(model, loader, optimizer, device: str, settings: dict, train: bool, scheduler=None) -> float:
    model.train(train)
    total = 0.0
    seen = 0
    accum = max(1, int(settings["gradient_accumulation_steps"]))
    step = 0
    if train:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader, start=1):
        batch = _move(batch, device)
        with torch.set_grad_enabled(train):
            _logits, loss = model(**batch)
            scaled = loss / accum
        if train:
            scaled.backward()
            if batch_index % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["max_grad_norm"]))
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % int(settings["logging_steps"]) == 0:
                    print(f"  step {step} loss {float(loss.detach()):.4f}", flush=True)
        total += float(loss.detach()) * batch["input_ids"].size(0)
        seen += batch["input_ids"].size(0)
    if train and seen % accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["max_grad_norm"]))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    return total / max(seen, 1)


class PageFiles(torch.utils.data.Dataset):
    """Encoded pages stored on disk so the full split does not have to stay in RAM."""

    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        return torch.load(self.paths[index], weights_only=False)


_PAGE_KEYS = ("input_ids", "attention_mask", "bbox", "pixel_values", "labels", "loss_mask")


def _slim_target(page: dict) -> dict:
    targets = page["targets"]
    return {
        "words": targets["words"],
        "multi_label_words": targets["multi_label_words"],
        "positive_words": targets["positive_words"],
        "unmatched_fields": targets["unmatched_fields"],
        "retention": targets["retention"],
        "labeled_truncated": len(page["labeled_truncated"]),
        "supervised_tokens": int(page["loss_mask"].sum()),
    }


def encode_to_disk(docs, processor, max_length: int, directory: Path) -> tuple[list[Path], dict]:
    """Write one tensor file per page. An interrupted encode continues from the saved pages."""
    directory.mkdir(parents=True, exist_ok=True)
    progress_path = directory / "pages.jsonl"
    saved: dict[str, dict] = {}
    if progress_path.is_file():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                saved[row["document_id"]] = row
    paths: list[Path] = []
    slims: list[dict] = []
    with progress_path.open("a", encoding="utf-8") as handle:
        for index, doc in enumerate(docs, start=1):
            existing = saved.get(doc.document_id)
            path = directory / f"{index:05d}.pt"
            if existing and Path(existing["file"]).is_file():
                paths.append(Path(existing["file"]))
                slims.append(existing["slim"])
            else:
                page = encode_document(doc, processor, max_length=max_length)
                torch.save({key: page[key] for key in _PAGE_KEYS}, path)
                slim = _slim_target(page)
                record = {"document_id": doc.document_id, "file": str(path), "slim": slim}
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                paths.append(path)
                slims.append(slim)
                del page
            if index == 1 or index % 50 == 0:
                gc.collect()
                print(f"  encoded {index}", flush=True)
    audit = audit_encoded_slims(slims)
    return paths, audit


def audit_encoded_slims(slims: list[dict]) -> dict:
    summary = pool_target_audit(slims)
    summary["labeled_words_truncated"] = sum(row["labeled_truncated"] for row in slims)
    summary["supervised_tokens"] = int(sum(row["supervised_tokens"] for row in slims))
    return summary


def encode_split(docs, processor, max_length: int) -> list[dict]:
    encoded = []
    for index, doc in enumerate(docs, start=1):
        encoded.append(encode_document(doc, processor, max_length=max_length))
        if index % 50 == 0:
            print(f"  encoded {index}", flush=True)
    return encoded


def _write_result(path: Path, payload: dict) -> None:
    assert_diagnostic_destination(path)
    write_json(path, payload)


def _trainer_state(epoch, history, best_loss, best_epoch, wait, stopped_early, settings, train_pages) -> dict:
    return {
        "epoch": epoch,
        "best_val_loss": best_loss,
        "best_epoch": best_epoch,
        "wait": wait,
        "patience": int(settings["patience"]),
        "min_delta": float(settings["min_delta"]),
        "stopped_early": stopped_early,
        "history": history,
        "train_pages": len(train_pages),
        "learning_rate": float(settings["learning_rate"]),
        "validation_used_for_gradients": False,
    }


def main() -> int:
    config_path = ROOT / "configs" / "multilabel-200.yaml"
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    settings = load_settings(config_path)
    try:
        work_dir, persistent_dir = resolve_run_dirs(settings)
    except RuntimeError as exc:
        print(exc, flush=True)
        return 1
    assert_softmax_checkpoint_untouched(work_dir)
    assert_softmax_checkpoint_untouched(Path(settings["output_dir"]))
    if persistent_dir is not None:
        assert_softmax_checkpoint_untouched(persistent_dir)
    try:
        device = describe_device(bool(settings.get("require_gpu", False)))
    except RuntimeError as exc:
        print(exc, flush=True)
        return 1
    _seed(int(settings["seed"]), device)
    print(f"Device: {device}", flush=True)
    print(f"Checkpoints: {work_dir}", flush=True)
    if persistent_dir is not None:
        print(f"Persistent copy: {persistent_dir}", flush=True)
    print(
        f"Train limit {settings['train_limit']} val {settings['eval_limit']} "
        f"epochs {settings['epochs']} batch {settings['batch_size']}",
        flush=True,
    )

    from transformers import LayoutLMv3Model, LayoutLMv3Processor

    processor = LayoutLMv3Processor.from_pretrained(settings["model_name"], apply_ocr=False)
    print("Loading pages", flush=True)
    train_limit = settings["train_limit"]
    resolved_limit = None if train_limit is None else int(train_limit)
    val_docs = prepare_documents(Path(settings["data_dir"]), "val", int(settings["eval_limit"]))
    print(f"Validation pages={len(val_docs)}", flush=True)
    print("Encoding multi-hot targets", flush=True)
    max_length = int(settings["max_seq_length"])
    if settings.get("encoded_cache"):
        from docintel.data.docile import iter_docile_pages

        train_paths, train_audit = encode_to_disk(
            iter_docile_pages(Path(settings["data_dir"]), "train", limit=resolved_limit),
            processor,
            max_length,
            work_dir / "encoded",
        )
        train_pages = PageFiles(train_paths)
        print(f"Train pages={len(train_pages)}", flush=True)
    else:
        train_docs = prepare_documents(Path(settings["data_dir"]), "train", resolved_limit)
        print(f"Train pages={len(train_docs)}", flush=True)
        train_pages = encode_split(train_docs, processor, max_length)
        train_audit = audit_encoded(train_pages)
        del train_docs
    val_pages = encode_split(val_docs, processor, max_length)
    del val_docs
    gc.collect()
    audit = {
        "train": {**train_audit, "pages": len(train_pages)},
        "validation": {**audit_encoded(val_pages), "pages": len(val_pages)},
    }
    diagnostics = ROOT / "artifacts" / "diagnostics"
    run_name = settings.get("run_name")
    targets_name = f"{run_name}-targets.json" if run_name else "multilabel-targets.json"
    _write_result(diagnostics / targets_name, audit)
    print(
        f"Train multi-label words {audit['train']['multi_label_words']} "
        f"unmatched fields {audit['train']['unmatched_fields']} "
        f"truncated labeled words {audit['train']['labeled_words_truncated']}",
        flush=True,
    )

    encoder = LayoutLMv3Model.from_pretrained(settings["model_name"])
    model = MultiLabelLayoutLM(encoder, num_labels=len(FIELD_NAMES)).to(device)
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

    work_dir.mkdir(parents=True, exist_ok=True)
    if settings.get("resume", False):
        restore_latest_from_persistent(work_dir, persistent_dir)
    try:
        planned = plan_resume(
            work_dir,
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
        load_training_checkpoint(model, planned["epoch_dir"], optimizer, scheduler, device)
        gc.collect()
        print(f"Continuing from epoch {start_epoch - 1}", flush=True)
    if stopped_early:
        print("Early stopping already reached", flush=True)
        start_epoch = int(settings["epochs"]) + 1
    log_path = work_dir / "train_log.jsonl"
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
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        write_json(work_dir / "trainer_state.json", state)
        print(f"Epoch {epoch} train {train_loss:.4f} val {val_loss:.4f}", flush=True)
        save_training_checkpoint(
            model,
            work_dir / f"epoch-{epoch}",
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
                work_dir / "best",
                field_names=list(FIELD_NAMES),
                model_name=settings["model_name"],
                optimizer=optimizer,
                scheduler=scheduler,
                processor=processor,
                trainer_state=state,
                extra=row,
            )
        sync_checkpoint(work_dir, persistent_dir, epoch, best_updated=best_updated)
        if stopped_early:
            print("Early stopping", flush=True)
            break

    summary = {
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "best_val_loss": best_loss,
        "best_epoch": best_epoch,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "history": history,
        "validation_used_for_gradients": False,
        "train_pages": len(train_pages),
        "initialization": settings["model_name"],
        "learning_rate": float(settings["learning_rate"]),
        "warmup_ratio": float(settings.get("warmup_ratio", 0.0)),
    }
    write_json(work_dir / "train_metrics.json", summary)
    del model, optimizer, scheduler, train_loader
    gc.collect()
    print("Evaluating best checkpoint", flush=True)
    best_dir = work_dir / "best"
    if not (best_dir / "model.pt").is_file() and history:
        chosen = min(history, key=lambda row: row["val_loss"])["epoch"]
        best_dir = work_dir / f"epoch-{chosen}"
    best_model, _processor, _payload = load_multilabel(best_dir, device)
    predicted = predict_pages(best_model, val_pages, settings, device)
    predicted["official"] = official_kile(settings, val_pages, predicted["predictions"])
    report = comparison_report(settings, predicted, summary)
    prefix = run_name or "multilabel-run"
    eval_name = f"{prefix}-eval.json"
    comparison_name = f"{prefix}-comparison.json"
    predictions_name = f"{prefix}-val-predictions.json"
    for directory in (diagnostics, work_dir / "eval"):
        _write_result(directory / predictions_name, predicted["predictions"])
        _write_result(directory / comparison_name, report)
        _write_result(directory / eval_name, predicted["official"])
    scored = predicted
    scored_pages = val_pages
    report_limit = settings.get("report_limit")
    if report_limit and int(report_limit) != len(val_pages):
        print(f"Scoring the same {report_limit} validation pages as the epoch-8 benchmark", flush=True)
        report_docs = prepare_documents(Path(settings["data_dir"]), "val", int(report_limit))
        scored_pages = encode_split(report_docs, processor, max_length)
        scored = predict_pages(best_model, scored_pages, settings, device)
        scored["official"] = official_kile(settings, scored_pages, scored["predictions"])
    benchmark_path = diagnostics / "multilabel-reconstruction-100.json"
    benchmark = {}
    if benchmark_path.is_file():
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8")).get("official", {})

    def _nine(block: dict, key: str):
        return ((block or {}).get(key) or {}).get("nine_trained_fields")

    comparison = {
        "benchmark": "epoch-8 + line-block-v2",
        "benchmark_checkpoint_updated": False,
        "threshold_changed": False,
        "grouping": settings.get("grouping"),
        "threshold": settings["threshold"],
        "full_model": str(best_dir),
        "validation_pages": len(scored_pages),
        "scorecard": extraction_scorecard(scored["official"]),
        "epoch8_scorecard": extraction_scorecard(benchmark) if benchmark else {},
        "nine_field_f1": {
            "epoch8_overlap_0.5": _nine(benchmark, "iou_0.5"),
            "full_overlap_0.5": _nine(scored["official"], "iou_0.5"),
            "epoch8_overlap_1.0": _nine(benchmark, "iou_1.0"),
            "full_overlap_1.0": _nine(scored["official"], "iou_1.0"),
            "epoch8_exact_text": _nine(benchmark, "iou_1.0_exact_text"),
            "full_exact_text": _nine(scored["official"], "iou_1.0_exact_text"),
        },
        "epoch8": benchmark,
        "full_training": scored["official"],
    }
    versus_name = f"{prefix}-vs-epoch8.json"
    for directory in (diagnostics, work_dir / "eval"):
        _write_result(directory / versus_name, comparison)
    if history:
        sync_checkpoint(work_dir, persistent_dir, history[-1]["epoch"], best_updated=True)
    print(f"Wrote {diagnostics / versus_name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
