"""Read rendered page images with docTR. Run this with .venv-doctr only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def read_image(engine, image_path: Path) -> list[dict]:
    image = Image.open(image_path).convert("RGB")
    try:
        result = engine([np.asarray(image)])
    finally:
        image.close()
    words = []
    for block in result.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                text = str(word.value).strip()
                if not text:
                    continue
                (x0, y0), (x1, y1) = word.geometry
                score = getattr(word, "confidence", None)
                words.append(
                    {
                        "text": text,
                        "box": [float(x0), float(y0), float(x1), float(y1)],
                        "ocr_confidence": None if score is None else float(score),
                    }
                )
    return words


def main() -> int:
    from doctr.models import ocr_predictor
    import doctr

    jobs_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
    engine = ocr_predictor(pretrained=True)
    pages = []
    for job in jobs:
        pages.append(
            {
                "page_index": int(job["page_index"]),
                "words": read_image(engine, Path(job["image"])),
            }
        )
    output_path.write_text(
        json.dumps(
            {
                "reader": "docTR",
                "version": getattr(doctr, "__version__", "unknown"),
                "detection": type(engine.det_predictor.model).__name__,
                "recognition": type(engine.reco_predictor.model).__name__,
                "pages": pages,
            }
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
