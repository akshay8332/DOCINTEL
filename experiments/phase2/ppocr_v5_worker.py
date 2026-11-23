"""Read invoice pages with PaddleOCR PP-OCRv5 and write relative word boxes.

Run this with .venv-paddle. The training environment must not import Paddle.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_onednn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from PIL import Image


def _numbers(points) -> list[float]:
    if hasattr(points, "tolist"):
        points = points.tolist()
    if isinstance(points, (int, float)):
        return [float(points)]
    values: list[float] = []
    for item in points:
        values.extend(_numbers(item))
    return values


def _relative_box(points, width: int, height: int) -> list[float] | None:
    values = _numbers(points)
    if len(values) < 4:
        return None
    if len(values) == 4:
        xs = [values[0], values[2]]
        ys = [values[1], values[3]]
    else:
        xs = values[0::2]
        ys = values[1::2]
    if not xs or not ys:
        return None
    return [
        max(0.0, min(1.0, min(xs) / width)),
        max(0.0, min(1.0, min(ys) / height)),
        max(0.0, min(1.0, max(xs) / width)),
        max(0.0, min(1.0, max(ys) / height)),
    ]


def _payload(result) -> dict:
    page = result[0] if isinstance(result, list) and result else result
    if hasattr(page, "json"):
        page = page.json
    if isinstance(page, dict) and "res" in page:
        page = page["res"]
    return page if isinstance(page, dict) else {}


def words_from_result(result, width: int, height: int) -> tuple[list[dict], str]:
    payload = _payload(result)
    lines = payload.get("text_word")
    boxes = payload.get("text_word_boxes") or payload.get("text_word_region")
    words: list[dict] = []
    if lines and boxes:
        for texts, polygons in zip(lines, boxes):
            if hasattr(texts, "tolist"):
                texts = texts.tolist()
            if hasattr(polygons, "tolist"):
                polygons = polygons.tolist()
            for text, polygon in zip(texts, polygons):
                cleaned = str(text).strip()
                box = _relative_box(polygon, width, height)
                if cleaned and box is not None:
                    words.append({"text": cleaned, "box": box})
        if words:
            return words, "word"
    for text, polygon in zip(payload.get("rec_texts") or [], payload.get("rec_polys") or []):
        cleaned = str(text).strip()
        box = _relative_box(polygon, width, height)
        if cleaned and box is not None:
            words.append({"text": cleaned, "box": box})
    return words, "line"


def _done(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    finished = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("crop_id") or row.get("document_id"):
            finished.add(row.get("crop_id") or row["document_id"])
    return finished


def main() -> None:
    jobs_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))["pages"]
    if limit is not None:
        jobs = jobs[:limit]
    finished = _done(out_path)
    from paddleocr import PaddleOCR

    engine = PaddleOCR(
        lang="en",
        ocr_version="PP-OCRv5",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        return_word_box=True,
        device="cpu",
        enable_mkldnn=False,
        cpu_threads=2,
    )
    params = getattr(engine, "_params", {})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not finished:
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "kind": "meta",
                        "reader": "PaddleOCR",
                        "ocr_version": "PP-OCRv5",
                        "text_detection_model_name": params.get("text_detection_model_name"),
                        "text_recognition_model_name": params.get("text_recognition_model_name"),
                        "return_word_box": True,
                    }
                )
                + "\n"
            )
    print(
        "models "
        f"{params.get('text_detection_model_name')} "
        f"{params.get('text_recognition_model_name')}",
        flush=True,
    )
    for index, job in enumerate(jobs, start=1):
        document_id = job.get("crop_id") or job["document_id"]
        if document_id in finished:
            print(f"skip {index}/{len(jobs)} {document_id}", flush=True)
            continue
        image = Image.open(job["image"])
        width, height = image.size
        image.close()
        result = engine.predict(job["image"], return_word_box=True)
        words, level = words_from_result(result, width, height)
        origin = job.get("origin")
        if origin is not None:
            x0, y0, x1, y1 = [float(value) for value in origin]
            span_x = x1 - x0
            span_y = y1 - y0
            words = [
                {
                    "text": word["text"],
                    "box": [
                        x0 + word["box"][0] * span_x,
                        y0 + word["box"][1] * span_y,
                        x0 + word["box"][2] * span_x,
                        y0 + word["box"][3] * span_y,
                    ],
                }
                for word in words
            ]
        row = {
            "document_id": job["document_id"],
            "crop_id": document_id,
            "words": words,
            "word_count": len(words),
            "level": level,
            "width": width,
            "height": height,
        }
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"read {index}/{len(jobs)} {document_id} words {len(words)} level {level}", flush=True)


if __name__ == "__main__":
    main()
