# Inference-time OCR plan

The frozen epoch-3 checkpoint stays unchanged. This plan only adds the missing step that turns a new PDF into the word boxes that checkpoint already expects.

The Colab notebook does not contain the preprocessing. It points at `configs/colab_full_training.yaml` and `experiments/multilabel/train.py`. The conventions below come from the code that training and the local audit actually call.

## What the model receives

`encode_document` in `experiments/multilabel/data.py` opens one rendered page and calls `LayoutLMv3Processor` with `apply_ocr=False`:

- `words`: one string per OCR word, in DocILE reading order
- `boxes`: one integer box per word, already on a 0–1000 scale
- the page image
- `truncation=True`, `padding="max_length"`, `max_length=512`

The installed tokenizer states that each box must already be normalized to 0–1000. It copies a word’s box onto that word’s subwords. It does not rescale pixels into boxes.

The checkpoint processor (`artifacts/checkpoints/epoch-3/processor_config.json`) then resizes the image to 224×224, resample mode 2, rescales by `1/255`, and normalizes with mean `0.5` and standard deviation `0.5`. That resize does not change the boxes.

Inference keeps words whose sigmoid probability is at least 0.5 and groups them with `line-block-v2`. The field score stored today is the mean of those word probabilities. It is not a calibrated probability and it is not an OCR score.

## DocILE OCR, measured from a file

`data/docile/ocr/009bdcdec5c04cd3b2e31555.json` has this shape:

- top level `pages`
- each page has `page_idx`, `dimensions`, and `blocks`
- `dimensions` for this page are `[1684, 1293]`, which is `[height, width]`
- the rendered PNG `009bdcdec5c04cd3b2e31555-p0.png` is 1293×1684 pixels, so width and height match that pair
- a word has `value`, `confidence`, `geometry`, and `snapped_geometry`
- `geometry` is `[[x0, y0], [x1, y1]]` with values in 0–1, top-left origin

`_page_words` walks `blocks`, then `lines`, then `words`. It keeps `value` and `geometry`. It does not pass `confidence` or `snapped_geometry` into the model.

`layout_box` converts a 0–1 box with `round(value * 1000)`, clamps each edge to 0–1000, and expands a zero-width or zero-height box by one unit. Training and the current audit both use that function.

`render_pdf_page` uses `pypdfium2` `page.render(scale=2.0)`, converts the bitmap to RGB, and saves a PNG. An existing non-empty PNG is reused.

## First OCR engine

New PDFs need a reader because DocILE’s OCR is a dataset file, not a program.

PP-OCRv5 and docTR both already have workers, and both stay outside the training virtual environment. A full-page PP-OCRv5 server detection pass on one 1293×1684 page exhausted this 5.8 GB machine and was stopped. docTR 1.1.0 ran on the address crops in `.venv-doctr`.

The first selectable provider is therefore docTR, called from `.venv-doctr`, on the scale-2 page image. Its worker already walks `page.blocks → lines → words` and records `word.geometry` as a 0–1 box, top-left origin. Those boxes then go through the existing `layout_box`. Word order is the order docTR returns. An engine score is stored only when docTR returns one. No score is filled in when it does not.

PP-OCRv5 remains a second provider behind the same interface. It is not the default on this machine. Reader agreement still runs only when two independent reader strings are present. docTR being the input reader does not count as one of those two.

## Reuse of a dataset OCR file

An upload may reuse `data/docile/ocr/{id}.json` only when all of these hold:

1. The file stem is a document id present in `train.json`, `val.json`, or `test.json`.
2. `data/docile/pdfs/{id}.pdf` and `data/docile/ocr/{id}.json` both exist.
3. The SHA-256 of the uploaded bytes equals the SHA-256 of that dataset PDF.

A matching filename with a different hash is a different document. That upload runs docTR and records that the name matched but the file did not.

## Processing stages

The current job has one status, `processing`. The upload job will move through:

`uploaded → rendered → ocr → inference → corrections → completed`

A failure stores the stage name and the message. The UI shows that stage. Original model text and corrected text stay in separate columns, as they do now.

Each run records:

- OCR provider name and version
- whether the words came from a verified DocILE file or from docTR
- checkpoint `docile_multilabel_full_5180/epoch-3`
- correction version `stack-v1:correct+boundary+fallback`
- which of those three layers changed the text
- duration

## Validation

The comparison uses the same first 100 validation pages and the official annotations. It writes a new report, `artifacts/diagnostics/ocr-pipeline-docTR-100.json`. It does not overwrite `address-text-audit-epoch3-local` or any frozen diagnostic.

Reported for the new reader, beside the DocILE-OCR baseline:

- word coverage against DocILE words
- detection precision and recall
- bounding-box IoU
- exact address match
- CER and WER
- time per page

The epoch-3 result with DocILE OCR is the baseline: vendor exact 33/92 and customer exact 28/70 before the correction stack. The corrected development count, 72 of 162, belongs to that DocILE-OCR path. It is not a result for docTR until this report is written.

## Tests before the 100-page run

- `layout_box` on a 0–1 corner pair, including swapped corners and a zero-area box
- a pixel polygon divided by the rendered width and height, then passed through `layout_box`
- page 2 of a multi-page file does not reuse page 1’s boxes
- a dataset PDF whose hash matches reuses the DocILE words
- a renamed or edited PDF does not
- reader agreement stays off when only docTR ran
