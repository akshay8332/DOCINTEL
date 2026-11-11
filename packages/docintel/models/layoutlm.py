"""LayoutLMv3 (preferred) / LayoutLMv2 load, fine-tune helpers, inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoProcessor

from docintel.data.funsd import FUNSD_BIO_LABELS, id_to_label
from docintel.data.transforms import LayoutLMExample, encode_example_for_layoutlm
from docintel.schemas.extraction import ExtractionResult, FieldPrediction


@dataclass
class LayoutLMBundle:
    model: Any
    processor: Any
    model_name: str
    label_list: list[str]
    id2label: dict[int, str]
    label2id: dict[str, int]


def resolve_model_name(
    preferred: str = "microsoft/layoutlmv3-base",
    fallback: str = "microsoft/layoutlmv2-base-uncased",
    prefer_v3: bool = True,
) -> str:
    """Return preferred LayoutLM name; fall back if import/config probe fails."""
    if not prefer_v3:
        return fallback
    try:
        AutoConfig.from_pretrained(preferred)
        return preferred
    except Exception:
        return fallback


def load_layoutlm_for_token_classification(
    model_name: str | None = None,
    *,
    label_list: list[str] | None = None,
    prefer_v3: bool = True,
    fallback_model_name: str = "microsoft/layoutlmv2-base-uncased",
    num_labels: int | None = None,
) -> LayoutLMBundle:
    labels = list(label_list or FUNSD_BIO_LABELS)
    name = model_name or resolve_model_name(
        preferred="microsoft/layoutlmv3-base",
        fallback=fallback_model_name,
        prefer_v3=prefer_v3,
    )
    n_labels = num_labels or len(labels)
    id2label = {i: lab for i, lab in enumerate(labels)}
    label2id = {lab: i for i, lab in enumerate(labels)}

    config = AutoConfig.from_pretrained(
        name,
        num_labels=n_labels,
        id2label=id2label,
        label2id=label2id,
    )
    # LayoutLMv3 uses AutoProcessor; v2 often uses AutoTokenizer + feature extractor.
    # AutoProcessor covers v3; for v2 we still try processor then tokenizer.
    try:
        processor = AutoProcessor.from_pretrained(name, apply_ocr=False)
    except Exception:
        from transformers import AutoTokenizer

        processor = AutoTokenizer.from_pretrained(name)

    model = AutoModelForTokenClassification.from_pretrained(name, config=config)
    return LayoutLMBundle(
        model=model,
        processor=processor,
        model_name=name,
        label_list=labels,
        id2label=id2label,
        label2id=label2id,
    )


def build_trainer_args(
    output_dir: str | Path,
    *,
    num_train_epochs: float = 5.0,
    max_steps: int | None = None,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    logging_steps: int = 10,
    eval_strategy: str = "epoch",
    save_strategy: str = "epoch",
    save_total_limit: int = 2,
    fp16: bool = False,
    seed: int = 42,
    dataloader_num_workers: int = 0,
) -> Any:
    import inspect

    from transformers import TrainingArguments

    kwargs: dict[str, Any] = dict(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        fp16=fp16,
        seed=seed,
        dataloader_num_workers=dataloader_num_workers,
        report_to=[],
        remove_unused_columns=False,
    )
    if max_steps is not None and max_steps > 0:
        kwargs["max_steps"] = max_steps

    # transformers>=5 removed warmup_ratio; map to warmup_steps when needed.
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "warmup_ratio" in ta_params:
        kwargs["warmup_ratio"] = warmup_ratio
    else:
        if max_steps is not None and max_steps > 0 and warmup_ratio > 0:
            kwargs["warmup_steps"] = max(1, int(warmup_ratio * max_steps))
        else:
            kwargs["warmup_steps"] = 0

    return TrainingArguments(**kwargs)


def _spans_from_bio(
    words: Sequence[str],
    labels: Sequence[str],
    confidences: Sequence[float],
    boxes: Sequence[list[int]] | None = None,
) -> list[FieldPrediction]:
    fields: list[FieldPrediction] = []
    i = 0
    n = len(words)
    while i < n:
        lab = labels[i]
        if lab == "O" or not lab.startswith("B-"):
            i += 1
            continue
        entity = lab[2:]
        texts = [words[i]]
        confs = [confidences[i]]
        start = i
        j = i + 1
        while j < n and labels[j] == f"I-{entity}":
            texts.append(words[j])
            confs.append(confidences[j])
            j += 1
        box = None
        if boxes is not None:
            span_boxes = boxes[start:j]
            box = [
                min(b[0] for b in span_boxes),
                min(b[1] for b in span_boxes),
                max(b[2] for b in span_boxes),
                max(b[3] for b in span_boxes),
            ]
        conf = sum(confs) / max(len(confs), 1)
        fields.append(
            FieldPrediction(
                label=entity,
                text=" ".join(texts),
                confidence=float(conf),
                box=box,
                start_token=start,
                end_token=j - 1,
            )
        )
        i = j
    return fields


@torch.inference_mode()
def infer_spans(
    bundle: LayoutLMBundle,
    example: LayoutLMExample,
    *,
    device: str | torch.device | None = None,
    max_length: int = 512,
    confidence_threshold: float = 0.75,
) -> ExtractionResult:
    """Run token classification and collapse BIO tags into field spans."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    bundle.model.to(device)
    bundle.model.eval()

    encoded = encode_example_for_layoutlm(
        example, bundle.processor, max_length=max_length
    )
    # Drop non-tensor keys
    tensor_inputs = {
        k: torch.tensor([v]).to(device)
        for k, v in encoded.items()
        if k in ("input_ids", "attention_mask", "bbox", "token_type_ids", "pixel_values")
    }
    outputs = bundle.model(**tensor_inputs)
    logits = outputs.logits[0]  # [seq, num_labels]
    probs = torch.softmax(logits, dim=-1)
    pred_ids = logits.argmax(dim=-1).tolist()
    conf_vals = probs.max(dim=-1).values.tolist()

    word_ids = None
    try:
        # Re-tokenize to get word_ids alignment (v3 processor needs images first).
        from docintel.data.transforms import _load_example_image, _processor_expects_images

        if _processor_expects_images(bundle.processor):
            tok = bundle.processor(
                _load_example_image(example),
                example.words,
                boxes=example.boxes,
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
        else:
            tok = bundle.processor(
                example.words,
                boxes=example.boxes,
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
        word_ids = tok.word_ids()
    except Exception:
        word_ids = None

    id2 = id_to_label(bundle.label_list)
    # Map first subword prediction back to each word
    word_labels = ["O"] * len(example.words)
    word_conf = [0.0] * len(example.words)
    if word_ids is not None:
        seen: set[int] = set()
        for pos, wid in enumerate(word_ids):
            if wid is None or wid in seen or wid >= len(example.words):
                continue
            seen.add(wid)
            word_labels[wid] = id2.get(pred_ids[pos], "O")
            word_conf[wid] = float(conf_vals[pos])
    else:
        for i, pid in enumerate(pred_ids[: len(example.words)]):
            word_labels[i] = id2.get(pid, "O")
            word_conf[i] = float(conf_vals[i])

    fields = _spans_from_bio(example.words, word_labels, word_conf, example.boxes)
    for f in fields:
        f.needs_review = f.confidence < confidence_threshold

    return ExtractionResult(
        document_id=example.document_id,
        fields=fields,
        model_name=bundle.model_name,
        raw_token_labels=word_labels,
    )
