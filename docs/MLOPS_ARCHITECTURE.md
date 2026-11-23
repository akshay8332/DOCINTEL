# MLOps architecture

MLflow records the frozen epoch-3 benchmark. It does not train, and it does not change weights, the 0.50 threshold, line-block-v2, or the correction stack.

## What is already recorded

`ProcessingRun` stores the model version, checkpoint reference, correction version, duration, and configuration. The configuration includes threshold, grouping, correction layers, OCR source, and OCR version. `ReviewHistory` stores the original prediction, the analyst value, and the same version strings.

That is the production audit trail. MLflow is the experiment register beside it. A prediction row should keep pointing at the same version the register calls the benchmark.

Fixed identity of the selected model:

| Item | Value |
|---|---|
| Weights | `artifacts/checkpoints/epoch-3` |
| Model version string | `layoutlmv3-multilabel-epoch3` |
| Checkpoint reference | `docile_multilabel_full_5180/epoch-3` |
| Architecture | LayoutLMv3 plus nine sigmoid heads |
| Threshold | 0.50 |
| Grouping | `line-block-v2` |
| Correction stack | `stack-v1:correct+boundary+fallback` |
| OCR for a new PDF | docTR, separate environment today |
| Raw exact | 61/162 |
| Corrected exact | 72/162 |
| Customer raw exact | 28/70 |
| False-positive words at that benchmark | 171 |

The party fine-tune at `artifacts/checkpoints/docile_party_experiment` is a completed negative result: 59/162 raw, 72/162 after the stack, customer raw 26/70. It can be logged as a rejected run. It is not registered as Production.

## Local MLflow

The Compose service `mlflow` is the tracking server. Open `http://localhost:5000`. Its database and artifacts stay in `artifacts/mlflow`, which is gitignored. No Databricks, SageMaker, or AWS account is connected.

`scripts/register_epoch3_benchmark.py` records the benchmark. Run it again with `docker compose exec mlflow python /register.py`. A second run does not create another benchmark version.

The registration step logs one model version:

- Parameters: threshold, grouping, correction version, seed 42, learning rate 2e-5, the nine field names.
- Metrics: the published 61/162, 72/162, and 28/70. These numbers are copied from the existing audit. They are not recomputed by a new training job.
- Artifact: the epoch-3 directory, or a pointer to its content hash, stored outside git.
- Tag: `stage=benchmark`.

No `mlflow run` starts `train.py`.

## What each prediction should carry

The database already has the version fields. A later change can add an MLflow model version id to `configuration`. Until that change is approved, the existing strings are the link.

For each invoice the stored facts are:

- Checkpoint version and correction version.
- OCR engine and the version string docTR returns.
- Threshold and grouping.
- Duration in seconds.
- Original text and corrected text.
- Analyst approved value, when a person saves one.

Inference latency is the duration already written on the run. Processing failures are `invoices.error_message`. Analyst corrections are rows in `review_history`.

## What MLflow is not asked to do

- Retrain epoch-3.
- Promote the party fine-tune.
- Replace SQLite or the review screen.
- Host the model on SageMaker.
- Score the public test set.

## Learning sequence

1. Start the local tracking server and open the UI.
2. Log the benchmark parameters and metrics by hand from the published audit.
3. Register the checkpoint as a model version.
4. Only after deploy, write the same version string the API already returns in `/health` into the MLflow description so the two stay the same.

Cloud tracking, if it is ever added, is an S3 artifact location plus a small tracking process. That has a cost note of its own and is not part of the first register.
