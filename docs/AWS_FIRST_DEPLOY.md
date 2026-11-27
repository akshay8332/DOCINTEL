# First AWS deploy: S3 upload + GitHub Actions → ECR → App Runner

This guide assumes you already created:

- one private **S3** bucket
- one **ECR** repository

It does **not** upgrade your account. App Runner bills for vCPU and memory while the service is running. Delete or pause it when you are not testing.

Epoch-3 weights, threshold 0.50, `line-block-v2`, and the correction stack stay unchanged.

---

## Part A — Upload the checkpoint to S3

The Docker image does **not** contain `model.pt`. App Runner needs the checkpoint in S3.

### A1. Confirm local files

On your PC, these must exist under `artifacts/checkpoints/epoch-3/`:

- `model.pt` (~501 MB)
- `tokenizer.json`
- `tokenizer_config.json`
- `processor_config.json`
- `trainer_state.json` (optional but useful)

### A2. Console upload (no AWS CLI required)

1. Open **S3** → your bucket.
2. Confirm **Block Public Access** is on.
3. Create folder: `models/epoch-3/`
4. Upload every file from `artifacts/checkpoints/epoch-3/` into that folder.
5. Final key example: `models/epoch-3/model.pt`

Cost: about **1¢/month** storage for 0.5 GB, plus tiny request charges.

### A3. Optional CLI upload (after `aws configure`)

```powershell
aws s3 sync .\artifacts\checkpoints\epoch-3 s3://YOUR_BUCKET/models/epoch-3/ --region YOUR_REGION
aws s3 ls s3://YOUR_BUCKET/models/epoch-3/
```

Replace `YOUR_BUCKET` and `YOUR_REGION`. Never commit access keys.

---

## Part B — GitHub secrets for Actions deploy

The workflow `.github/workflows/deploy-ecr.yml` is **manual only**. It builds the API image and pushes it to ECR. It does not create buckets or start App Runner by itself.

### B1. Create an IAM user for deploy (console)

1. IAM → Users → Create user (example name: `docintel-github-deploy`).
2. Attach a policy that allows ECR push to **your** repository only, for example:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:CompleteLayerUpload",
        "ecr:InitiateLayerUpload",
        "ecr:PutImage",
        "ecr:UploadLayerPart",
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer",
        "ecr:DescribeRepositories"
      ],
      "Resource": "arn:aws:ecr:YOUR_REGION:YOUR_ACCOUNT_ID:repository/YOUR_ECR_REPO"
    }
  ]
}
```

3. Create an access key for that user. Copy the key id and secret once.

### B2. Add GitHub repository secrets

GitHub → your repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**:

| Secret name | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM access key id |
| `AWS_SECRET_ACCESS_KEY` | IAM secret access key |
| `AWS_REGION` | e.g. `us-east-1` |
| `ECR_REPOSITORY` | ECR repo name only, e.g. `docintel-api` |

Do not put these values in `.env` or commit them.

---

## Part C — Deploy the image from GitHub Actions

1. Push the latest `main` (includes the deploy workflow and S3 checkpoint fetch).
2. GitHub → **Actions** → **Deploy API image to ECR**.
3. Click **Run workflow**.
4. Leave tag as `latest` (or set a version tag).
5. Wait. The first build downloads CPU PyTorch and can take 20–45 minutes.
6. When green, open ECR → your repo → confirm an image tagged `latest`.

Image URI shape:

```text
YOUR_ACCOUNT.dkr.ecr.YOUR_REGION.amazonaws.com/YOUR_ECR_REPO:latest
```

---

## Part D — App Runner Express Mode

### D1. Cost reminder

App Runner charges roughly:

- **$0.064 per vCPU-hour**
- **$0.007 per GB-hour**

Example: 1 vCPU + 2 GB left on all month ≈ **$50+**. Use the smallest size, and delete the service when idle. Prefer **deploy from ECR image**, not “build from GitHub” (avoids App Runner build minutes and the $1 automatic-deploy add-on when possible).

### D2. Create the service

1. App Runner → Create (Express Mode is fine).
2. Source: **Container registry** → **Amazon ECR** → select the image you pushed.
3. Port: `8080`.
4. CPU / memory: smallest that can load the model (try **1 vCPU / 2 GB** first; raise only if the service OOMs).
5. Environment variables:

| Name | Value |
|---|---|
| `DOCINTEL_DEVICE` | `cpu` |
| `DOCINTEL_INFERENCE_MODE` | `auto` |
| `DOCINTEL_FROZEN_CHECKPOINT` | `epoch-3` |
| `DOCINTEL_CHECKPOINT_DIR` | `/models` |
| `DOCINTEL_S3_CHECKPOINT_URI` | `s3://YOUR_BUCKET/models/epoch-3` |
| `DOCINTEL_STORAGE_ROOT` | `/tmp/uploads` |
| `DOCINTEL_INVOICE_DB_URL` | `sqlite:////tmp/docintel.db` |

6. Instance role / access role: the App Runner instance needs `s3:ListBucket` and `s3:GetObject` on `YOUR_BUCKET/models/epoch-3/*`. Attach a role with that access so startup can download the checkpoint.
7. Create the service. First start downloads ~0.5 GB from S3, then loads the model. Health may take a few minutes.

### D3. Verify

Open:

```text
https://YOUR_APPRUNNER_URL/health
```

Expect:

- `inference_mode`: `real`
- `checkpoint_present`: `true`
- `threshold`: `0.5`
- `grouping`: `line-block-v2`
- `model_version`: `layoutlmv3-multilabel-epoch3`

Then upload a small PDF through the API (`POST /api/invoices/upload` and `POST /api/invoices/{id}/process`).

### D4. Frontend later

The Vue app can stay local, or you upload `apps/web/dist` to S3 static hosting after setting `VITE_API_URL` to the App Runner URL and rebuilding. That is a separate step.

---

## Part E — Day-2 loop

| Change | What to run |
|---|---|
| Code / Dockerfile change | Actions → **Deploy API image to ECR** → then App Runner redeploy / new deployment |
| New checkpoint (not epoch-3) | Do not replace epoch-3 unless you explicitly decide to; upload under a new S3 prefix |
| Stop spending | App Runner → delete or pause the service |

---

## Checklist

- [ ] S3 private bucket with `models/epoch-3/model.pt`
- [ ] GitHub secrets set
- [ ] Actions deploy pushed an image to ECR
- [ ] App Runner service uses that image
- [ ] Instance role can read the S3 prefix
- [ ] `/health` shows `real` and epoch-3
- [ ] Service deleted or paused when not testing
