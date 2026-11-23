"""Read address crops with docTR and write relative word boxes.

Run this with .venv-doctr. The training environment does not import docTR.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


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


def _words(page) -> list[dict]:
    words = []
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                text = str(word.value).strip()
                (x0, y0), (x1, y1) = word.geometry
                if text:
                    words.append(
                        {
                            "text": text,
                            "box": [float(x0), float(y0), float(x1), float(y1)],
                        }
                    )
    return words


def _to_page(words: list[dict], origin: list[float] | None) -> list[dict]:
    if origin is None:
        return words
    x0, y0, x1, y1 = [float(value) for value in origin]
    span_x = x1 - x0
    span_y = y1 - y0
    return [
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


def main() -> None:
    jobs_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))["pages"]
    if limit is not None:
        jobs = jobs[:limit]
    finished = _done(out_path)
    from doctr.models import ocr_predictor

    engine = ocr_predictor(pretrained=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not finished:
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "kind": "meta",
                        "reader": "docTR",
                        "ocr_version": "pretrained",
                        "text_detection_model_name": type(engine.det_predictor.model).__name__,
                        "text_recognition_model_name": type(engine.reco_predictor.model).__name__,
                    }
                )
                + "\n"
            )
    print(
        "models "
        f"{type(engine.det_predictor.model).__name__} "
        f"{type(engine.reco_predictor.model).__name__}",
        flush=True,
    )
    for index, job in enumerate(jobs, start=1):
        crop_id = job.get("crop_id") or job["document_id"]
        if crop_id in finished:
            print(f"skip {index}/{len(jobs)} {crop_id}", flush=True)
            continue
        image = Image.open(job["image"]).convert("RGB")
        result = engine([np.asarray(image)])
        words = _to_page(_words(result.pages[0]), job.get("origin"))
        image.close()
        row = {
            "document_id": job["document_id"],
            "crop_id": crop_id,
            "words": words,
            "word_count": len(words),
            "level": "word",
        }
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"read {index}/{len(jobs)} {crop_id} words {len(words)}", flush=True)


if __name__ == "__main__":
    main()
