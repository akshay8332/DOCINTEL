"""CLI: fine-tune LayoutLM on a limited number of DocILE invoice pages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune LayoutLM on DocILE invoices")
    parser.add_argument(
        "--config",
        type=Path,
        default=_repo_root() / "configs" / "training-invoice.yaml",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=200, help="Training page images")
    parser.add_argument("--eval-limit", type=int, default=20, help="Validation page images")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda|cpu|auto")
    args = parser.parse_args(argv)

    from docintel.data.docile import INVOICE_BIO_LABELS, load_docile_pages
    from docintel.data.transforms import encode_example_for_layoutlm, prepare_examples
    from docintel.models.layoutlm import (
        build_trainer_args,
        load_layoutlm_for_token_classification,
    )
    from docintel.models.metrics import align_predictions, entity_f1_report
    from docintel.schemas.config import TrainingConfig

    cfg = TrainingConfig.from_yaml(args.config)
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.epochs is not None:
        cfg.num_train_epochs = args.epochs
    if args.batch_size is not None:
        cfg.per_device_train_batch_size = args.batch_size
    if not cfg.label_list or cfg.label_list == ["O"]:
        cfg.label_list = list(INVOICE_BIO_LABELS)

    device_opt = (args.device or "auto").lower()
    if device_opt == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_opt
    print(f"Device: {device}")
    print(
        f"Config: model={cfg.model_name} batch={cfg.per_device_train_batch_size} "
        f"epochs={cfg.num_train_epochs} limit={args.limit} eval_limit={args.eval_limit}"
    )

    train_docs = load_docile_pages(cfg.data_dir, "train", limit=args.limit)
    eval_docs = load_docile_pages(cfg.data_dir, "val", limit=args.eval_limit)
    if not train_docs:
        print("No DocILE train pages found.", file=sys.stderr)
        return 1
    print(f"Pages: train={len(train_docs)} val={len(eval_docs)}")

    train_examples = prepare_examples(train_docs, label_list=cfg.label_list)
    eval_examples = prepare_examples(eval_docs, label_list=cfg.label_list)

    bundle = load_layoutlm_for_token_classification(
        cfg.model_name,
        label_list=cfg.label_list,
        prefer_v3=cfg.prefer_v3,
        fallback_model_name=cfg.fallback_model_name,
    )
    print(f"Loaded model: {bundle.model_name}")

    def _encode_many(examples):
        rows = []
        for ex in examples:
            enc = encode_example_for_layoutlm(
                ex, bundle.processor, max_length=cfg.max_seq_length
            )
            enc.pop("document_id", None)
            rows.append(enc)
        return rows

    from datasets import Dataset
    from transformers import Trainer, default_data_collator

    print("Encoding pages...")
    train_ds = Dataset.from_list(_encode_many(train_examples))
    eval_ds = Dataset.from_list(_encode_many(eval_examples)) if eval_examples else None

    training_args = build_trainer_args(
        cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy if eval_ds is not None else "no",
        save_strategy=cfg.save_strategy,
        save_total_limit=cfg.save_total_limit,
        fp16=cfg.fp16 and device == "cuda",
        seed=cfg.seed,
        dataloader_num_workers=cfg.dataloader_num_workers,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        y_true, y_pred = align_predictions(preds, labels, bundle.id2label)
        report = entity_f1_report(y_true, y_pred)
        return {
            "f1_micro": report["f1_micro"],
            "f1_macro": report["f1_macro"],
            "precision_micro": report["precision_micro"],
            "recall_micro": report["recall_micro"],
        }

    trainer = Trainer(
        model=bundle.model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if cfg.eval_strategy != "no" else None,
        data_collator=default_data_collator,
        processing_class=bundle.processor,
        compute_metrics=compute_metrics if eval_ds is not None else None,
    )

    train_result = trainer.train()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(cfg.output_dir))
    bundle.processor.save_pretrained(str(cfg.output_dir))

    metrics_path = Path(cfg.output_dir) / "train_metrics.json"
    payload = {
        "train": train_result.metrics,
        "model_name": bundle.model_name,
        "device": device,
        "train_pages": len(train_docs),
        "eval_pages": len(eval_docs),
    }
    if eval_ds is not None and cfg.eval_strategy != "no":
        payload["eval"] = trainer.evaluate()
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved checkpoint -> {cfg.output_dir}")
    print(f"Metrics -> {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
