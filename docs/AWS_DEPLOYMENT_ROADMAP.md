# AWS deployment roadmap

Each step keeps epoch-3, threshold 0.50, line-block-v2, and `stack-v1:correct+boundary+fallback`. No step creates AWS resources until the cost note for that step is accepted and the console shows the service is available on this account. The Free account plan blocks many billable services. If the console asks for a paid plan, stop.

## Step 0. Record the benchmark

Purpose: the deployed model is the one already measured.

The selected checkpoint is `artifacts/checkpoints/epoch-3`. Benchmark on the labelled 100-page set: raw exact 61/162, corrected exact 72/162, customer raw exact 28/70. The party fine-tune is recorded in `docs` and `artifacts/diagnostics/party-experiment-summary.md` and is not deployed.

Learning point: a model version is the weights plus the threshold, grouping, OCR, and correction stack. Changing any of those is a different version.

## Step 1. Local Docker Compose

Purpose: prove the same extraction in a clean container before AWS.

Local files are in place. How to run them is in `docs/LOCAL_DOCKER.md`.

- `docker/Dockerfile.api` installs CPU PyTorch, the API, `experiments/multilabel`, and docTR.
- `docker/Dockerfile.web` builds the Vue app and nginx proxies `/api`.
- `.dockerignore` keeps datasets, checkpoints, virtual environments, and `*.pt` out of the build context.
- Compose mounts `artifacts/checkpoints/epoch-3` read-only and stores SQLite and uploads in a volume.
- `docker/Dockerfile.worker` is still the old training sketch. Compose does not start it.

Exit test, still to be confirmed by a running container: a PDF returns vendor and customer addresses from epoch-3, and the run row stores `layoutlmv3-multilabel-epoch3` and `stack-v1:correct+boundary+fallback`.

## Step 2. MLflow on the laptop

Purpose: register epoch-3 without training.

The Compose service `mlflow` is running. The UI is `http://localhost:5000`. The registered model `docile-address-epoch3` alias `benchmark` points at epoch-3, with the published metrics and the fixed threshold, grouping, and correction version. The party fine-tune is a rejected run. It is not the benchmark.

## Step 3. GitHub Actions for tests and image build

Purpose: see CI before any deploy.

The workflow is `.github/workflows/ci.yml`.

- On every push and pull request: Python tests (CPU PyTorch) and `npm run build`.
- After those pass: build the web Docker image from the uploaded `dist` artifact.
- On `main` pushes and manual `workflow_dispatch`: also build the API Docker image.
- Images are not pushed to ECR yet. That waits for the AWS account check in Step 4.

Learning point: build and deploy are different. A green test job does not mean a server changed.

## Step 4. Account check, still no resources

Purpose: see what this Free plan can create.

In the console, open S3, ECR, App Runner, and CloudWatch. If a service says it requires the paid plan, leave it. Do not click upgrade. Write down the result in this file before Step 5.

## Step 5. First paid-or-free-tier cloud slice

Purpose: one public API and a static site, only if Step 4 allows the services.

Order:

1. S3 bucket for PDFs, private, block public access. Application access through an IAM role.
2. ECR repository and one image tag.
3. One CPU API service, minimum size, no GPU.
4. S3 or the same pattern for the Vue build, with the API URL baked in at build time.
5. CloudWatch logs, retention 7 days.

SQLite can stay on the task for this slice if the service is a single copy. Moving to RDS is Step 7 and has its own cost note.

Exit test: the same PDF used in Step 1 returns the same address strings. `/health` shows epoch-3 and threshold 0.50.

## Step 6. Durable jobs

Purpose: a restart must not lose an in-flight invoice.

Replace the daemon thread with a queue. On AWS that queue is SQS, not ElastiCache. Locally it can be Redis and a worker so the learning exercise still covers a broker. The worker runs the same `InvoiceService.run_job` path.

## Step 7. PostgreSQL, only after the cost note

Purpose: a database that survives replacing the API task.

RDS PostgreSQL is not part of the first slice. The application already uses SQLAlchemy, so the URL can change later. There is no migration history. Creating the schema is `create_all` today. A migration tool is added in this step if RDS is approved, so later columns are not applied by hand.

## Step 8. Monitoring that matches the product

Purpose: see failures without a second platform.

Log inference duration, OCR source, processing failures, and review decisions. The duration is already stored on `processing_runs`. CloudWatch metric filters can count failures and latency from those logs. Analyst correction counts come from `review_history`. No new model metric replaces the 61/162 and 72/162 benchmark.

## What this roadmap refuses

- Launching EC2, GPU, NAT, or RDS during Steps 0–6.
- Putting AWS keys in the repo or the image.
- Retraining, or deploying `docile_party_experiment`.
- Treating a health check of `unavailable` as a successful deployment.
