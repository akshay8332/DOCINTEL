"""Load the first validation pages into the local app.

The PDFs and DocILE OCR already on disk are copied into the upload folder.
The model is not run. Open an invoice in the UI and choose Run extraction.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))

from docintel.data.docile import load_docile_pages
from docintel.database.session import make_session_factory
from docintel.schemas.config import AppSettings
from docintel.services.address_pipeline import AddressPipeline
from docintel.services.invoice_service import InvoiceService


def main() -> int:
    settings = AppSettings(max_upload_bytes=80 * 1024 * 1024)
    data_dir = settings.data_dir / "docile"
    pages = load_docile_pages(
        data_dir,
        "val",
        limit=100,
        image_dir=data_dir / "page_images",
    )
    grouped: dict[str, list] = defaultdict(list)
    for page in pages:
        doc_id = page.document_id.rsplit("-p", 1)[0]
        grouped[doc_id].append(page)

    service = InvoiceService(make_session_factory(settings.invoice_db_url), settings, AddressPipeline(settings))
    db = service.session()
    existing = {row.filename for row in service.list_invoices(db)}
    imported = 0
    skipped = 0
    try:
        for doc_id, doc_pages in grouped.items():
            filename = f"{doc_id}.pdf"
            if filename in existing:
                skipped += 1
                continue
            pdf_path = data_dir / "pdfs" / filename
            content = pdf_path.read_bytes()
            row = service.accept_upload(db, filename, content)
            sidecar = service.upload_root / f"{row.stored_name}.words.json"
            payload = {
                "pages": [
                    {
                        "page_index": int(page.document_id.rsplit("-p", 1)[1]),
                        "words": [
                            {"text": word.text, "box": [int(value) for value in word.box]}
                            for word in page.pages[0].words
                        ],
                    }
                    for page in sorted(doc_pages, key=lambda item: item.document_id)
                ]
            }
            sidecar.write_text(json.dumps(payload), encoding="utf-8")
            imported += 1
    finally:
        db.close()
    print(f"Imported {imported} validation invoices. Already present: {skipped}. Pages read: {len(pages)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
