"""Load the same DocILE pages as the softmax run and encode multi-hot labels."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import yaml

from multilabel import FIELD_NAMES
from multilabel.align import align_word_targets
from multilabel.targets import pool_target_audit, word_targets


def _resolve_setting_path(root: Path, value: str) -> str:
    text = str(value).replace("\\", "/")
    if text.startswith("/content"):
        return text
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((root / path).resolve())


def load_settings(path: Path) -> dict:
    settings = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = Path(path).resolve().parents[1]
    for key in (
        "data_dir",
        "output_dir",
        "softmax_report",
        "weights_checkpoint",
        "scratch_dir",
        "persistent_dir",
    ):
        if key in settings and settings[key]:
            settings[key] = _resolve_setting_path(root, settings[key])
    for key, env_name in (
        ("data_dir", "DOCINTEL_DATA_DIR"),
        ("docile_src", "DOCINTEL_DOCILE_SRC"),
        ("persistent_dir", "DOCINTEL_PERSISTENT_DIR"),
        ("scratch_dir", "DOCINTEL_SCRATCH_DIR"),
    ):
        if os.environ.get(env_name):
            settings[key] = os.environ[env_name]
    settings["docile_src"] = str(Path(settings["docile_src"]))
    return settings


def assert_softmax_checkpoint_untouched(output_dir: Path) -> None:
    """Refuse to overwrite a frozen baseline checkpoint."""
    from multilabel.checkpoint import assert_output_allowed

    assert_output_allowed(output_dir)


def prepare_documents(data_dir: Path, split: str, limit: int | None):
    from docintel.data.docile import load_docile_pages

    return load_docile_pages(data_dir, split, limit=limit)


def _token_tensor(value) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.long)
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor


def _box_tensor(value) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.long)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor


def encode_document(doc, processor, *, max_length: int = 512) -> dict:
    from PIL import Image

    targets = word_targets(doc)
    words = [word.text for word in doc.pages[0].words]
    boxes = [list(word.box) for word in doc.pages[0].words]
    image = Image.open(doc.pages[0].image_path).convert("RGB")
    encoding = processor(
        image,
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    image.close()
    aligned = align_word_targets(targets["matrix"], list(encoding.word_ids()), len(FIELD_NAMES))
    pixels = encoding["pixel_values"]
    if isinstance(pixels, (list, tuple)) and len(pixels) == 1:
        pixels = pixels[0]
    pixel_values = torch.as_tensor(pixels, dtype=torch.float32)
    if pixel_values.ndim == 4:
        pixel_values = pixel_values.squeeze(0)
    page_index = int(doc.document_id.rsplit("-p", 1)[1])
    labeled_truncated = [
        index
        for index in aligned["truncated_words"]
        if any(targets["matrix"][index])
    ]
    return {
        "input_ids": _token_tensor(encoding["input_ids"]),
        "attention_mask": _token_tensor(encoding["attention_mask"]),
        "bbox": _box_tensor(encoding["bbox"]),
        "pixel_values": pixel_values,
        "labels": torch.tensor(aligned["labels"], dtype=torch.float32),
        "loss_mask": torch.tensor(aligned["loss_mask"], dtype=torch.float32),
        "document_id": doc.document_id,
        "page_index": page_index,
        "words": words,
        "boxes": boxes,
        "word_ids": list(encoding.word_ids()),
        "targets": targets,
        "labeled_truncated": labeled_truncated,
    }


def collate_pages(batch: list[dict]) -> dict:
    keys = ("input_ids", "attention_mask", "bbox", "pixel_values", "labels", "loss_mask")
    return {key: torch.stack([item[key] for item in batch]) for key in keys}


def audit_encoded(pages: list[dict]) -> dict:
    summary = pool_target_audit([page["targets"] for page in pages])
    summary["labeled_words_truncated"] = sum(len(page["labeled_truncated"]) for page in pages)
    summary["supervised_tokens"] = int(sum(page["loss_mask"].sum().item() for page in pages))
    return summary


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
