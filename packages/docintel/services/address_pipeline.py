"""Invoice address inference.

This module calls the frozen multi-label LayoutLMv3 path and the existing
correction functions. It does not define new correction rules, change the
0.5 threshold, or change line-block-v2 grouping.
"""

from __future__ import annotations

import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

from docintel.schemas.config import AppSettings

ADDRESS_FIELDS = ("vendor_address", "customer_billing_address")
CORRECTION_VERSION = "stack-v1:correct+boundary+fallback"
CORRECTION_LAYERS = ("correct", "boundary", "fallback")
THRESHOLD = 0.5
GROUPING = "line-block-v2"
MODEL_VERSION = "layoutlmv3-multilabel-epoch3"
CHECKPOINT_REFERENCE = "docile_multilabel_full_5180/epoch-3"
SCORE_MEANING = (
    "Mean probability of the grouped words. "
    "This is not the probability that the address text is correct."
)

_LAYER_FUNCTIONS = {}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_experiments() -> None:
    experiments = repo_root() / "experiments"
    entry = str(experiments)
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _layer_functions():
    if _LAYER_FUNCTIONS:
        return _LAYER_FUNCTIONS
    _ensure_experiments()
    from multilabel.post_ocr import correct_prediction, refine_boundary, selective_fallback

    _LAYER_FUNCTIONS["correct"] = correct_prediction
    _LAYER_FUNCTIONS["boundary"] = refine_boundary
    _LAYER_FUNCTIONS["fallback"] = selective_fallback
    return _LAYER_FUNCTIONS


def checkpoint_dir(settings: AppSettings) -> Path:
    return (settings.checkpoint_dir / settings.frozen_checkpoint).resolve()


def checkpoint_ready(settings: AppSettings) -> bool:
    folder = checkpoint_dir(settings)
    return (folder / "model.pt").is_file()


def resolved_mode(settings: AppSettings) -> str:
    requested = (settings.inference_mode or "auto").strip().lower()
    if requested == "mock":
        return "mock"
    if checkpoint_ready(settings):
        return "real"
    return "unavailable"


def _words(text: str) -> list[str]:
    return (text or "").replace("\n", " ").split()


def _layers_for_final(original: str, steps: list[tuple[str, str]]) -> dict[int, str | None]:
    current = _words(original)
    layers: list[str | None] = [None] * len(current)
    for name, text in steps:
        nxt = _words(text)
        matcher = SequenceMatcher(a=current, b=nxt, autojunk=False)
        updated: list[str | None] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                updated.extend(layers[i1:i2])
            else:
                updated.extend([name] * (j2 - j1))
        current = nxt
        layers = updated
    return {index: layer for index, layer in enumerate(layers)}


def _removal_layers(original: str, steps: list[tuple[str, str]]) -> list[tuple[str, str]]:
    recorded: list[tuple[str, str]] = []
    current = _words(original)
    for name, text in steps:
        nxt = _words(text)
        matcher = SequenceMatcher(a=current, b=nxt, autojunk=False)
        for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
            if tag in {"delete", "replace"}:
                recorded.extend((word, name) for word in current[i1:i2])
        current = nxt
    return recorded


def word_diff(
    original: str,
    corrected: str,
    layers: dict[int, str | None] | None = None,
    removals: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """Compare two address strings. A layer is set only when a real step changed that token."""
    left = _words(original)
    right = _words(corrected)
    known = layers or {}
    pending = list(removals or [])
    items: list[dict] = []

    def take_removed(word: str) -> str | None:
        for index, (text, layer) in enumerate(pending):
            if text == word:
                pending.pop(index)
                return layer
        return None

    matcher = SequenceMatcher(a=left, b=right, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            items.extend({"text": word, "change": "unchanged", "layer": None} for word in left[i1:i2])
        elif tag == "delete":
            items.extend(
                {"text": word, "change": "removed", "layer": take_removed(word)} for word in left[i1:i2]
            )
        elif tag == "insert":
            for offset, word in enumerate(right[j1:j2]):
                items.append({"text": word, "change": "added", "layer": known.get(j1 + offset)})
        else:
            items.extend(
                {"text": word, "change": "removed", "layer": take_removed(word)} for word in left[i1:i2]
            )
            for offset, word in enumerate(right[j1:j2]):
                change = "changed" if i2 - i1 == j2 - j1 else "added"
                items.append({"text": word, "change": change, "layer": known.get(j1 + offset)})
    return items


def apply_existing_stack(original: str, reader_texts: list[str] | None = None) -> dict:
    """Run the existing three layers and record which layer changed the string."""
    functions = _layer_functions()
    readers = [text for text in (reader_texts or []) if text]
    readers_ran = len(readers) >= 2
    current = original or ""
    steps: list[tuple[str, str]] = []
    applied: list[str] = []
    for name in CORRECTION_LAYERS:
        if name == "fallback" and not readers_ran:
            continue
        updated = functions[name](current) if name != "fallback" else functions[name](current, readers)
        if updated != current:
            applied.append(name)
        steps.append((name, updated))
        current = updated
    return {
        "original_text": original or "",
        "corrected_text": current,
        "layers_applied": applied,
        "readers_ran": readers_ran,
        "token_changes": word_diff(
            original or "",
            current,
            _layers_for_final(original or "", steps),
            _removal_layers(original or "", steps),
        ),
    }


class AddressPipeline:
    """Lazy loader around the existing checkpoint. One instance is reused across invoices."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._model = None

    def status(self) -> dict:
        mode = resolved_mode(self.settings)
        return {
            "status": "ok",
            "inference_mode": mode,
            "demo": mode == "mock",
            "checkpoint_present": checkpoint_ready(self.settings),
            "checkpoint_reference": CHECKPOINT_REFERENCE,
            "model_version": MODEL_VERSION if mode == "real" else None,
            "correction_version": CORRECTION_VERSION,
            "threshold": THRESHOLD,
            "grouping": GROUPING,
        }

    def process_invoice(
        self,
        file_path: Path,
        *,
        words_path: Path | None = None,
        filename: str = "",
        on_stage=None,
    ) -> dict:
        mode = resolved_mode(self.settings)
        started = time.perf_counter()
        if mode == "mock":
            result = _demo_result()
        elif mode == "unavailable":
            raise InferenceUnavailable(_missing_checkpoint_message())
        else:
            result = self._run_frozen(file_path, words_path, filename, on_stage)
        result["duration_seconds"] = round(time.perf_counter() - started, 4)
        result["model_version"] = MODEL_VERSION if mode == "real" else "demo"
        result["checkpoint_reference"] = CHECKPOINT_REFERENCE if mode == "real" else None
        result["correction_version"] = CORRECTION_VERSION
        result["configuration"] = {
            "threshold": THRESHOLD,
            "grouping": GROUPING,
            "layers": list(CORRECTION_LAYERS),
            "mode": mode,
            "readers_ran": any(field.get("readers_ran") for field in result["fields"]),
            "warnings": result.get("warnings") or [],
            "ocr_source": result.get("ocr_source"),
            "ocr_reader": result.get("ocr_reader"),
            "ocr_version": result.get("ocr_version"),
        }
        return result

    def _load(self):
        if self._model is not None:
            return self._model
        _ensure_experiments()
        from multilabel.evaluate import load_multilabel

        folder = checkpoint_dir(self.settings)
        if not (folder / "model.pt").is_file():
            raise InferenceUnavailable(_missing_checkpoint_message())
        device = "cpu" if self.settings.device in {"auto", "cpu"} else self.settings.device
        model, processor, payload = load_multilabel(folder, device)
        self._model = (model, processor, payload, device)
        return self._model

    def _run_frozen(self, file_path: Path, words_path: Path | None, filename: str, on_stage) -> dict:
        from docintel.services.ocr_source import render_pages, resolve_words

        def stage(name: str) -> None:
            if on_stage is not None:
                on_stage(name)

        stage("rendered")
        render_pages(file_path)
        stage("ocr")
        if words_path is not None and not Path(str(file_path) + ".words.json").is_file():
            Path(str(file_path) + ".words.json").write_bytes(words_path.read_bytes())
        resolved = resolve_words(file_path, filename or file_path.name, self.settings.data_dir / "docile")
        pages = _read_words(Path(str(file_path) + ".words.json"))
        stage("inference")
        model, processor, _payload, device = self._load()
        encoded = [_encode_page(processor, file_path, page) for page in pages]
        _ensure_experiments()
        from multilabel.evaluate import predict_pages

        predicted = predict_pages(
            model,
            encoded,
            {"threshold": THRESHOLD, "grouping": GROUPING},
            device,
        )
        by_field: dict[str, list] = {name: [] for name in ADDRESS_FIELDS}
        for document_fields in predicted["predictions"].values():
            for item in document_fields:
                if item["fieldtype"] in by_field:
                    by_field[item["fieldtype"]].append(item)
        reader_map = pages[0].get("reader_texts") if pages else {}
        if not isinstance(reader_map, dict):
            reader_map = {}
        stage("corrections")
        fields = []
        warnings = [
            "Reader agreement runs only when two independent reader strings are available for that field."
        ]
        if resolved.get("source") == "doctr":
            warnings.append("Words were read by docTR. This is not the DocILE OCR used to train the model.")
        for name in ADDRESS_FIELDS:
            candidates = by_field[name]
            if not candidates:
                fields.append(_empty_field(name))
                continue
            chosen = max(candidates, key=lambda item: float(item.get("score") or 0))
            readers = reader_map.get(name) or []
            if not isinstance(readers, list):
                readers = []
            corrected = apply_existing_stack(chosen.get("text") or "", [str(item) for item in readers])
            if not corrected["readers_ran"]:
                warnings.append(f"Reader agreement was not run for {name}.")
            box = [float(value) for value in chosen.get("bbox") or []]
            fields.append(
                {
                    "field_name": name,
                    "original_text": corrected["original_text"],
                    "corrected_text": corrected["corrected_text"],
                    "page_index": int(chosen.get("page") or 0),
                    "page_number": int(chosen.get("page") or 0) + 1,
                    "bounding_box": box or None,
                    "box_0_1000": [round(value * 1000) for value in box] if box else None,
                    "layers_applied": corrected["layers_applied"],
                    "readers_ran": corrected["readers_ran"],
                    "token_changes": corrected["token_changes"],
                    "confidence_signals": {
                        "model_score": float(chosen.get("score") or 0),
                        "calibrated_probability": None,
                        "score_meaning": SCORE_MEANING,
                        "demo": False,
                        "located": True,
                        "readers_ran": corrected["readers_ran"],
                    },
                    "demo": False,
                }
            )
        if len(candidates := by_field["vendor_address"]) > 1 or len(by_field["customer_billing_address"]) > 1:
            warnings.append("More than one address span was found. The highest mean word score is shown.")
        return {
            "mode": "real",
            "fields": fields,
            "warnings": warnings,
            "page_count": len(pages),
            "ocr_source": resolved.get("source"),
            "ocr_reader": resolved.get("reader"),
            "ocr_version": resolved.get("version"),
        }


class InferenceUnavailable(RuntimeError):
    """The frozen model cannot run. The message is safe to show to a reviewer."""


def _missing_checkpoint_message() -> str:
    return (
        "The frozen full-training checkpoint "
        f"{CHECKPOINT_REFERENCE} is not installed. "
        "The 200-page pilot weights are a different model and are not used."
    )


def _empty_field(name: str) -> dict:
    blank = apply_existing_stack("")
    return {
        "field_name": name,
        "original_text": "",
        "corrected_text": "",
        "page_index": None,
        "page_number": None,
        "bounding_box": None,
        "box_0_1000": None,
        "layers_applied": [],
        "readers_ran": False,
        "token_changes": blank["token_changes"],
        "confidence_signals": {
            "model_score": None,
            "calibrated_probability": None,
            "score_meaning": SCORE_MEANING,
            "demo": False,
            "located": False,
            "readers_ran": False,
            "note": "The model did not mark this field at threshold 0.5.",
        },
        "demo": False,
    }


def _demo_result() -> dict:
    samples = {
        "vendor_address": (
            "DEMO ORIGINAL VENDOR\n100 MAIN STREET",
            "DEMO CORRECTED VENDOR\n100 MAIN STREET",
            [0.08, 0.08, 0.48, 0.2],
        ),
        "customer_billing_address": (
            "DEMO ORIGINAL CUSTOMER\n200 MARKET STREET",
            "DEMO CORRECTED CUSTOMER\n200 MARKET STREET",
            [0.52, 0.08, 0.92, 0.2],
        ),
    }
    fields = []
    for name, (original, corrected, box) in samples.items():
        fields.append(
            {
                "field_name": name,
                "original_text": original,
                "corrected_text": corrected,
                "page_index": 0,
                "page_number": 1,
                "bounding_box": box,
                "box_0_1000": [int(value * 1000) for value in box],
                "layers_applied": [],
                "readers_ran": False,
                "token_changes": word_diff(original, corrected),
                "confidence_signals": {
                    "model_score": None,
                    "calibrated_probability": None,
                    "score_meaning": None,
                    "demo": True,
                    "located": False,
                    "readers_ran": False,
                    "note": "Demo mode is on. These values are not model predictions.",
                },
                "demo": True,
                "box_note": "Demo highlight. This box was not produced by the frozen model.",
            }
        )
    return {
        "mode": "mock",
        "fields": fields,
        "warnings": ["Demo mode is on. Nothing on this invoice came from the frozen checkpoint."],
        "page_count": None,
    }


def _read_words(path: Path) -> list[dict]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    pages = payload.get("pages") if isinstance(payload, dict) else None
    if not isinstance(pages, list) or not pages:
        raise InferenceUnavailable("The words file has no pages.")
    cleaned = []
    for page in pages:
        words = page.get("words") or []
        if not words:
            raise InferenceUnavailable("A page in the words file has no words.")
        cleaned.append(
            {
                "page_index": int(page.get("page_index") or 0),
                "words": words,
                "reader_texts": page.get("reader_texts") or payload.get("reader_texts") or {},
            }
        )
    return cleaned


def _encode_page(processor, pdf_path: Path, page: dict) -> dict:
    """Processor inputs only. Gold annotations are not read."""
    import torch
    from PIL import Image

    from docintel.data.docile import render_pdf_page

    page_index = int(page["page_index"])
    image_path = pdf_path.parent.parent / "processed" / pdf_path.stem / f"p{page_index}.png"
    render_pdf_page(pdf_path, page_index, image_path, scale=2.0)
    words = [str(item["text"]) for item in page["words"]]
    boxes = [list(item["box"]) for item in page["words"]]
    image = Image.open(image_path).convert("RGB")
    try:
        encoding = processor(
            image,
            words,
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=512,
        )
    finally:
        image.close()
    pixels = encoding["pixel_values"]
    if isinstance(pixels, (list, tuple)) and len(pixels) == 1:
        pixels = pixels[0]
    pixel_values = torch.as_tensor(pixels, dtype=torch.float32)
    if pixel_values.ndim == 4:
        pixel_values = pixel_values.squeeze(0)

    def _tokens(value):
        tensor = torch.as_tensor(value, dtype=torch.long)
        if tensor.ndim == 2 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        return tensor

    bbox = _tokens(encoding["bbox"])
    return {
        "input_ids": _tokens(encoding["input_ids"]),
        "attention_mask": _tokens(encoding["attention_mask"]),
        "bbox": bbox,
        "pixel_values": pixel_values,
        "document_id": f"upload-p{page_index}",
        "page_index": page_index,
        "words": words,
        "boxes": boxes,
        "word_ids": list(encoding.word_ids()),
    }
