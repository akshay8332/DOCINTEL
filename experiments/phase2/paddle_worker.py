"""Run PaddleOCR in the separate environment and write relative word boxes.

The training environment does not import Paddle. This worker is executed with
.venv-paddle so NumPy there cannot change the checkpoint environment.
"""

from __future__ import annotations

import os

# Paddle 3.3 on this CPU raises an unimplemented oneDNN conversion unless
# oneDNN is turned off before Paddle is imported.
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_onednn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

import json
import sys
from pathlib import Path

from PIL import Image


def _polygon_box(points, width: int, height: int) -> list[float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [
        max(0.0, min(1.0, min(xs) / width)),
        max(0.0, min(1.0, min(ys) / height)),
        max(0.0, min(1.0, max(xs) / width)),
        max(0.0, min(1.0, max(ys) / height)),
    ]


def _pairs_from_result(result, width: int, height: int) -> list[dict]:
    words = []
    pages = result if isinstance(result, list) else [result]
    for page in pages:
        payload = page
        if hasattr(page, "json"):
            try:
                payload = page.json
            except Exception:
                payload = page
        if isinstance(payload, dict) and "res" in payload:
            payload = payload["res"]
        texts = None
        polys = None
        if isinstance(payload, dict):
            texts = payload.get("rec_texts") or payload.get("texts")
            polys = payload.get("rec_polys") or payload.get("dt_polys") or payload.get("rec_boxes")
        if texts is not None and polys is not None:
            for text, poly in zip(texts, polys):
                if not str(text).strip():
                    continue
                words.append({"text": str(text), "box": _polygon_box(poly, width, height)})
            continue
        # PaddleOCR 2.x: [[box, (text, score)], ...]
        lines = page if isinstance(page, list) else []
        if lines and isinstance(lines[0], list) and lines[0] and isinstance(lines[0][0], list):
            lines = lines[0] if lines and isinstance(lines[0][0], list) else lines
        for line in lines or []:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue
            poly, rec = line[0], line[1]
            text = rec[0] if isinstance(rec, (list, tuple)) else str(rec)
            if str(text).strip():
                words.append({"text": str(text), "box": _polygon_box(poly, width, height)})
    return words


def main() -> None:
    spec_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    jobs = json.loads(spec_path.read_text(encoding="utf-8"))
    from paddleocr import PaddleOCR

    engine = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    pages = []
    for job in jobs:
        image = Image.open(job["image"])
        width, height = image.size
        image.close()
        result = engine.predict(job["image"]) if hasattr(engine, "predict") else engine.ocr(job["image"])
        pages.append(
            {
                "document_id": job["document_id"],
                "words": _pairs_from_result(result, width, height),
            }
        )
    out_path.write_text(json.dumps(pages), encoding="utf-8")


if __name__ == "__main__":
    main()
