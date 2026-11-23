"""Find word boxes for one uploaded PDF.

A verified DocILE file is reused. Any other PDF is read with docTR in its own
virtual environment. Boxes are then scaled with the same layout_box used in training.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from docintel.data.docile import _page_words, layout_box, render_pdf_page
from docintel.services.address_pipeline import InferenceUnavailable, repo_root

DOCTR_SOURCE = "doctr"
DOCILE_SOURCE = "docile-file"
SIDECAR_SOURCE = "existing-sidecar"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scale_relative_box(box: list[float]) -> list[int]:
    """Turn a 0–1 top-left box into the 0–1000 box the frozen model expects."""
    return layout_box([float(value) for value in box])


def matching_docile_document(filename: str, pdf_path: Path, data_dir: Path) -> str | None:
    """Return a dataset id only when the name and the file bytes both match."""
    stem = Path(filename).stem.lower()
    if len(stem) != 24 or any(char not in "0123456789abcdef" for char in stem):
        return None
    listed = False
    for split in ("train", "val", "test"):
        split_path = data_dir / f"{split}.json"
        if not split_path.is_file():
            continue
        ids = json.loads(split_path.read_text(encoding="utf-8"))
        if stem in ids:
            listed = True
            break
    dataset_pdf = data_dir / "pdfs" / f"{stem}.pdf"
    dataset_ocr = data_dir / "ocr" / f"{stem}.json"
    if not listed or not dataset_pdf.is_file() or not dataset_ocr.is_file():
        return None
    if sha256_file(pdf_path) != sha256_file(dataset_pdf):
        return None
    return stem


def words_from_docile_ocr(ocr_path: Path) -> list[dict]:
    payload = json.loads(ocr_path.read_text(encoding="utf-8"))
    pages = []
    for page in payload.get("pages") or []:
        words = []
        for text, box in _page_words(page):
            entry = {"text": text, "box": scale_relative_box(box)}
            words.append(entry)
        pages.append({"page_index": int(page.get("page_idx") or 0), "words": words})
    if not pages or any(not page["words"] for page in pages):
        raise InferenceUnavailable("The matching DocILE OCR file has a page with no words.")
    return pages


def render_pages(pdf_path: Path) -> list[Path]:
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(pdf_path))
    try:
        count = len(document)
    finally:
        document.close()
    images = []
    folder = pdf_path.parent.parent / "processed" / pdf_path.stem
    for page_index in range(count):
        images.append(render_pdf_page(pdf_path, page_index, folder / f"p{page_index}.png", scale=2.0))
    return images


def doctr_interpreter() -> Path:
    """Python that has docTR.

    The laptop uses `.venv-doctr`. A container sets `DOCINTEL_DOCTR_PYTHON`
    to the interpreter inside the image. The OCR script is unchanged.
    """
    configured = os.environ.get("DOCINTEL_DOCTR_PYTHON", "").strip()
    candidates = []
    if configured:
        candidates.append(Path(configured))
    root = repo_root()
    candidates.append(root / ".venv-doctr" / "Scripts" / "python.exe")
    candidates.append(root / ".venv-doctr" / "bin" / "python")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise InferenceUnavailable("docTR is not installed in .venv-doctr, so this new PDF cannot be read.")


def read_with_doctr(image_paths: list[Path]) -> dict:
    python = doctr_interpreter()
    script = repo_root() / "scripts" / "doctr_read_pages.py"
    with tempfile.TemporaryDirectory() as folder:
        jobs = Path(folder) / "jobs.json"
        output = Path(folder) / "words.json"
        jobs.write_text(
            json.dumps(
                [
                    {"page_index": index, "image": str(path)}
                    for index, path in enumerate(image_paths)
                ]
            ),
            encoding="utf-8",
        )
        try:
            completed = subprocess.run(
                [str(python), str(script), str(jobs), str(output)],
                capture_output=True,
                text=True,
                timeout=1200,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise InferenceUnavailable("docTR did not finish reading this PDF.") from exc
        if completed.returncode != 0 or not output.is_file():
            raise InferenceUnavailable("docTR could not read this PDF.")
        payload = json.loads(output.read_text(encoding="utf-8"))
    pages = []
    for page in payload.get("pages") or []:
        words = []
        for word in page.get("words") or []:
            words.append({"text": str(word["text"]), "box": scale_relative_box(word["box"])})
        if not words:
            raise InferenceUnavailable(f"docTR returned no words on page {int(page.get('page_index') or 0) + 1}.")
        pages.append({"page_index": int(page.get("page_index") or 0), "words": words})
    return {
        "source": DOCTR_SOURCE,
        "reader": payload.get("reader") or "docTR",
        "version": payload.get("version"),
        "detection": payload.get("detection"),
        "recognition": payload.get("recognition"),
        "pages": pages,
    }


def resolve_words(pdf_path: Path, filename: str, data_dir: Path) -> dict:
    """Use a saved sidecar, a verified DocILE file, or docTR. The model is not called here."""
    sidecar = Path(str(pdf_path) + ".words.json")
    if sidecar.is_file():
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        return {
            "source": payload.get("source") or SIDECAR_SOURCE,
            "reader": payload.get("reader"),
            "version": payload.get("version"),
            "pages": payload.get("pages") or [],
        }
    matched = matching_docile_document(filename, pdf_path, data_dir)
    if matched:
        pages = words_from_docile_ocr(data_dir / "ocr" / f"{matched}.json")
        result = {"source": DOCILE_SOURCE, "reader": "docile", "version": None, "pages": pages}
    else:
        images = render_pages(pdf_path)
        result = read_with_doctr(images)
    sidecar.write_text(json.dumps(result), encoding="utf-8")
    return result
