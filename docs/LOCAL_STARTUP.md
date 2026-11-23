# Run DOCILE locally

The interface is `apps/web`. The API is the existing FastAPI app in `packages/docintel`. Model code stays in `experiments/multilabel`. Nothing here retrains the model or edits the correction rules.

## Versions on this machine

- Python 3.13.3. The project accepts Python 3.10 or newer.
- Node.js 24.19.0.
- The training environment is `f:\Projects\docintel\.venv` (CPU PyTorch). Do not install PaddleOCR into it. Paddle, docTR, and Surya keep their own environments.

## Backend

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev]"
python -m uvicorn docintel.api.main:app --host 127.0.0.1 --port 8080
```

OpenAPI is at `http://127.0.0.1:8080/docs`.

The database is created on startup at `data/app/docintel.db`. Uploads go to `data/uploads`. Both directories are gitignored. There is no separate migration tool; SQLAlchemy creates the tables.

## Frontend

```powershell
cd apps\web
npm install
npm run dev
```

The UI is at `http://127.0.0.1:5173` and proxies `/api` to port 8080.

## Environment

Copy `.env.example` to `.env` if you want overrides. Names use the `DOCINTEL_` prefix.

| Variable | Default | Meaning |
| --- | --- | --- |
| `DOCINTEL_INFERENCE_MODE` | `auto` | `auto` uses the frozen checkpoint when `model.pt` is present. Otherwise processing fails with a clear error. `mock` returns labeled demo addresses for the interface. `real` never substitutes demo data. |
| `DOCINTEL_CHECKPOINT_DIR` | `./artifacts/checkpoints` | Folder that contains checkpoint runs. |
| `DOCINTEL_FROZEN_CHECKPOINT` | `epoch-3` | Relative path of the verified full-training epoch-3 checkpoint. |
| `DOCINTEL_INVOICE_DB_URL` | `sqlite:///./data/app/docintel.db` | Invoice database. |
| `DOCINTEL_STORAGE_ROOT` | `./data/uploads` | Uploaded PDFs. |
| `DOCINTEL_MAX_UPLOAD_BYTES` | `20971520` | 20 MB. |
| `DOCINTEL_DEVICE` | `auto` | `auto` and `cpu` both run inference on CPU here. |
| `DOCINTEL_PROCESS_INLINE` | `false` | `true` finishes a job inside the request. Leave this false for the UI. Tests set it true. |

`DOCINTEL_CONFIDENCE_THRESHOLD` belongs to the older softmax stub and is not used by address extraction. The address threshold stays 0.5.

## Checkpoint

Real extraction uses the verified full-training epoch-3 folder:

`artifacts/checkpoints/epoch-3/`

`model.pt` in that folder records epoch 3 and validation loss 0.006091364833991975. It also needs the LayoutLMv3 processor files saved beside it. The local folders `docile_multilabel_200/epoch-3` and `docile_multilabel_200_baseline` are the 200-page pilot. The application does not load them.

A new PDF also needs DocILE-format word boxes, because the model was trained with its own OCR turned off. Place a sidecar next to the stored PDF named `{invoice-id}.pdf.words.json`:

```json
{
  "pages": [
    {
      "page_index": 0,
      "words": [{ "text": "ACME", "box": [10, 20, 80, 40] }]
    }
  ]
}
```

`box` is integers in 0–1000, top-left origin, `[x0, y0, x1, y1]`. Optional `reader_texts` can include the two comparison readers for one field:

```json
"reader_texts": {
  "vendor_address": ["text from docTR", "text from PP-OCRv5"]
}
```

Reader agreement does nothing unless both strings are present. The application does not launch the Paddle or docTR environments by itself.

The page image is rendered with the existing `pypdfium2` helper at scale 2. Poppler is not required for that path.

## What the screen is allowed to claim

- Demo mode is labeled. Its addresses and highlight boxes are fixtures.
- A model score, when real mode returns one, is the mean word probability. It is not a calibrated probability that the address is correct.
- Development counts from the 100 validation invoices are not shown as the accuracy of an upload.
- A match percentage appears only after you type reference text on the Evaluation page, and the response says it is not a model accuracy score.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/api/test_health_stub.py tests/api/test_invoices.py tests/unit/test_address_pipeline.py
```

## Troubleshooting

- **Processing fails immediately with “checkpoint is not installed.”** The full-training weights are not in `artifacts/checkpoints/epoch-3`. Set `DOCINTEL_INFERENCE_MODE=mock` to click through the UI, or copy that checkpoint into place. Do not point this setting at the 200-page pilot.
- **Processing fails asking for word boxes.** The PDF arrived without a `.words.json` sidecar. The frozen model cannot OCR a PDF itself.
- **The API is offline in the browser.** Start uvicorn on port 8080 before `npm run dev`.
- **A review seems to have changed the model output.** The original and corrected columns stay as the processing run wrote them. The human value is `approved_value`, and each action is appended to review history.
- **Paddle or docTR errors during a normal upload.** They are not on the upload path. A sidecar `reader_texts` entry is only consumed by the existing fallback function.

## Later MLOps

Not built in this pass: MLflow, a model registry, dataset versioning, drift monitors, Celery, or a production image. The seams that can take those later are:

- `AddressPipeline.process_invoice`, which is the only inference entry the API calls.
- `processing_runs.model_version`, `checkpoint_reference`, `correction_version`, and `configuration`.
- The in-process thread behind `InvoiceService.start_processing`, which can be replaced by a queue without changing the routes.
- `infra/compose.yml` and `infra/ci.yml`, which remain sketches.
