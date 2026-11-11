#!/usr/bin/env python
"""Evaluate a saved LayoutLM checkpoint on FUNSD test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "packages"))

import torch  # noqa: E402
from transformers import AutoModelForTokenClassification, AutoProcessor  # noqa: E402

from docintel.data.funsd import FUNSD_BIO_LABELS, load_funsd_split  # noqa: E402
from docintel.data.transforms import encode_example_for_layoutlm, prepare_examples  # noqa: E402
from docintel.models.metrics import align_predictions, entity_f1_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval FUNSD checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Directory with saved model + processor",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_ROOT / "data" / "funsd",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="Optional doc cap for quick eval")
    args = parser.parse_args()

    docs = load_funsd_split(args.data_dir, "test")
    if args.limit:
        docs = docs[: args.limit]
    examples = prepare_examples(docs, label_list=FUNSD_BIO_LABELS)

    processor = AutoProcessor.from_pretrained(args.checkpoint, apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    all_preds: list[list[int]] = []
    all_labels: list[list[int]] = []

    for ex in examples:
        enc = encode_example_for_layoutlm(ex, processor, max_length=args.max_length)
        labels = enc["labels"]
        inputs = {
            k: torch.tensor([v]).to(device)
            for k, v in enc.items()
            if k in ("input_ids", "attention_mask", "bbox", "token_type_ids", "pixel_values")
        }
        with torch.inference_mode():
            logits = model(**inputs).logits[0]
        preds = logits.argmax(dim=-1).tolist()
        all_preds.append(preds)
        all_labels.append(labels)

    y_true, y_pred = align_predictions(all_preds, all_labels, id2label)
    report = entity_f1_report(y_true, y_pred)
    out = args.checkpoint / "eval_metrics.json"
    out.write_text(json.dumps({k: v for k, v in report.items() if k != "detail"}, indent=2), encoding="utf-8")
    print(report["detail"])
    print(f"Wrote {out}")
    print(f"micro-F1={report['f1_micro']:.4f} macro-F1={report['f1_macro']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
