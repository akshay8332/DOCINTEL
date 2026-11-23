# DOCILE technical assessment

This note records the repository as it exists before the local invoice application was added. It does not change model weights, correction rules, notebooks, datasets, or saved validation outputs.

The product name in this repository is DocIntel. The dataset and the address task are DocILE. The application UI uses the name DOCILE.

## What the frozen system is

The address model is a multi-label LayoutLMv3 classifier, not the earlier softmax BIO tagger.

- Encoder: `microsoft/layoutlmv3-base`, `apply_ocr=False`.
- Head: `LayoutLMv3Model` plus `Linear(768, 9)` on text tokens only. Image patch tokens are dropped before the classifier. See `experiments/multilabel/model.py`.
- Nine fields, in order: `document_id`, `date_issue`, `date_due`, `vendor_name`, `customer_billing_name`, `vendor_address`, `customer_billing_address`, `amount_due`, `amount_total_gross`.
- Loss during training: masked binary cross-entropy on the first subword of each OCR word.
- Inference grouping: `line-block-v2` (`experiments/multilabel/group.py`, `INFERENCE_BASELINE`).
- Inference threshold: `0.5` for every field.
- The application is limited to the two address fields. The other seven fields stay in the model and are not shown as the product surface.

The selected full-training checkpoint is epoch 3 of the run `docile_multilabel_full_5180`, chosen by validation loss on the first 20 validation pages. Early stopping fired after epoch 5. The weights were trained from `microsoft/layoutlmv3-base`, not from the 200-page epoch-8 weights.

The verified weights are in `artifacts/checkpoints/epoch-3`. `model.pt` records epoch 3 and validation loss 0.006091364833991975.

| Path | What it is | Present locally |
| --- | --- | --- |
| `artifacts/checkpoints/epoch-3/model.pt` | Frozen full-training epoch 3 | Yes |
| Google Drive `DOCINTEL_DATA_DIR/artifacts/checkpoints/docile_multilabel_full_5180/best` | Same weights, saved from Colab | Not mounted |
| Kaggle dataset `akshayathreyas/docile-model/best` | A copy of that `best` folder | Not mounted |
| `artifacts/checkpoints/docile_multilabel_200/epoch-3` | A different, 200-page pilot. Its selected epoch was 8 | Yes |
| `artifacts/checkpoints/docile_multilabel_200_baseline` | Frozen copy of the 200-page epoch-8 run | Yes |
| `artifacts/checkpoints/invoice-layoutlm-200` | Earlier softmax model | Yes |
| `artifacts/checkpoints/docile_multilabel_full` | Interrupted local CPU encode cache, not weights | Encoded pages only |

The application must not load the 200-page epoch-3 folder as a substitute for the full-training checkpoint.

`model.pt` contains `encoder`, `classifier`, `field_names`, and `model_name`. The processor files saved beside it (`LayoutLMv3Processor.from_pretrained`) are also required. Optimizer files are not required for inference.

## How inference works

There is no notebook that is the live inference path. Training was run from `scripts/train_colab.py` and `notebooks/colab_full_training.ipynb`. Scoring uses `experiments/multilabel/infer.py` and `experiments/multilabel/evaluate.py`.

1. A page is a rendered PNG plus DocILE OCR words. `packages/docintel/data/docile.py` rasterizes with `pypdfium2` at scale 2 and reuses `data/docile/page_images/{doc}-p{index}.png`.
2. OCR words come from `data/docile/ocr/{doc}.json`. Boxes in that file are 0–1, stored as `[[x0, y0], [x1, y1]]`. `layout_box` scales them to integers in 0–1000.
3. `encode_document` in `experiments/multilabel/data.py` runs `LayoutLMv3Processor` with `apply_ocr=False`. It also reads gold annotations, so that function is a training/evaluation encoder, not a standalone reader of a new PDF.
4. `predict_pages` runs the model, takes a sigmoid, keeps words at probability ≥ 0.5, and calls `group_fields` with `line-block-v2`.
5. Each prediction exports:
   - `text`
   - `bbox`: four floats in 0–1, top-left origin, `[x0, y0, x1, y1]`
   - `page`: zero-based page index
   - `score`: mean probability of the grouped words. This is not a calibrated probability that the address string is correct.
   - `box_0_1000` exists on the Python object and is the union of word boxes before dividing by 1000.

DocILE’s own OCR is precomputed for the dataset. This repository does not contain a general OCR engine that turns an arbitrary uploaded PDF into the word boxes the frozen model was trained on. PaddleOCR PP-OCRv5 and docTR live in `.venv-paddle` and `.venv-doctr`. They were comparison readers. They are not the model’s input, and replacing DocILE OCR with either of them lost exact-match accuracy on the 100 validation invoices.

A newly uploaded PDF therefore cannot run through the frozen model until both of these exist locally:

- `artifacts/checkpoints/epoch-3` (`model.pt` plus the processor files)
- word boxes in the DocILE OCR geometry for that file

Until then, real processing must fail with that reason. Demo mode is a separate, labeled path for the interface.

## Correction stack

The rules live in `experiments/multilabel/post_ocr.py` and `experiments/multilabel/ocr_text.py`. They were not changed for the application.

Order, versioned as `stack-v1:correct+boundary+fallback`:

1. Character fixes: `P.0.` before `BOX` becomes `P.O.`, `I`/`l`/`O` inside a long digit-like token, a whole token `clo` becomes `c/o`, then spacing is tightened.
2. Boundary cleanup: drop a consecutive duplicate line, or a single leading digit. Company lines stay.
3. Reader agreement: replace one token only when docTR and PP-OCRv5 agree and the edit distance is 1 or 2, or split one token when both readers agree and the letters stay the same.

Reader agreement does nothing when those two word lists are absent. The application records that the layer was not run. It does not invent agreement.

Development results on the 100 validation invoices (162 gold addresses), kept as history, not as a live accuracy figure:

| | Exact | Wording |
| --- | ---: | ---: |
| Before correction | 61 | 70 |
| After the stack | 72 | 87 |

Word-edit check on that same set: 31 edits moved closer to the official wording, 3 were neutral, 0 moved farther. A later confidence model on this same set did not find a slice that stays at 95% precision. Those numbers must not be shown as the accuracy of an uploaded invoice.

## What was already stubbed

- `packages/docintel/api/main.py` served `/health` and empty `/v1/extract/sync` and `/v1/review` routes.
- `packages/docintel/services/extract.py` wraps the older softmax LayoutLM span decoder. That is not the address model.
- `configs/app.yaml` points the checkpoint at `funsd-layoutlm` and sets a review threshold of 0.75. The address pipeline does not use that threshold.
- `apps/monitor` is a Vue placeholder with no pages. The new UI is `apps/web`. The placeholder is left in place.
- SQLite, invoice tables, and a review workflow did not exist.
- `infra/compose.yml` and `infra/ci.yml` are sketches. They are not a running deployment.

## Coordinates

| Space | Meaning |
| --- | --- |
| OCR JSON | 0–1, two corner points |
| Model word boxes | integers 0–1000, top-left origin |
| Exported field box | 0–1, `[x0, y0, x1, y1]`, top-left origin |
| Page index | 0-based in the model; the UI shows page index + 1 |

## Dependencies already used by the ML path

Python 3.10+ (this machine: 3.13.3). Node on this machine: 24.19.0.

Training environment: `f:\Projects\docintel\.venv` with PyTorch, Transformers, FastAPI, Pydantic v2. Paddle, docTR, and Surya each have their own virtual environment and must stay out of the training environment.

System packages used when rendering dataset pages: `pypdfium2`. Poppler is not required for that renderer. The public DocILE evaluator needs `pdf2image` and Poppler, and it is not part of the upload path.

## What the application is allowed to do

- Call the existing loader, grouping, and correction functions.
- Store original text and corrected text separately.
- Fail closed when the frozen checkpoint or DocILE word boxes are missing.
- Run a labeled demo mode for the interface.

## What it must not do

- Retrain, or copy the 200-page weights into the frozen name.
- Add correction rules or change 0.5 / `line-block-v2`.
- Treat a model score as a calibrated probability.
- Compute accuracy on an upload that has no reference annotation.
- Show 61 / 72 or 70 / 87 as the accuracy of a new invoice.

## Application boundary

The local app added after this inspection calls the functions above. It does not reimplement them.

- Inference entry: `packages/docintel/services/address_pipeline.py`. Real mode loads `artifacts/checkpoints/epoch-3` through `load_multilabel` and `predict_pages`. If that checkpoint or a DocILE word sidecar is missing, the job is stored as failed.
- `DOCINTEL_INFERENCE_MODE=mock` is the only demo path. Demo rows are flagged `demo: true` and carry no model score.
- Correction version stored on every run: `stack-v1:correct+boundary+fallback`.
- Startup steps: `docs/LOCAL_STARTUP.md`.
