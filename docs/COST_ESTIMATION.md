# Cost estimation

These are planning ranges, not a bill and not a quote. Prices move. Check the AWS Pricing Calculator for the chosen Region before creating anything. Amounts below are on the order of US East (N. Virginia) public prices in 2026.

The Free account plan is not the same thing as the 12-month free tier. The Free plan is built to block services that can charge the account. EC2, NAT gateways, RDS, and GPU instances are in that category. Do not upgrade the plan from this project. If the console requires a paid plan, stop and decide explicitly.

A free-tier allowance on a paid account is a monthly or 12-month credit. Usage past the allowance is billed. A service with a free tier can still cost money.

## Stay at zero

Local Docker Compose, local SQLite, the local epoch-3 folder, and a local MLflow file store cost nothing on AWS. That is the development setup. GitHub Actions minutes for a private repository are included up to the GitHub Free quota. A Docker build of PyTorch can consume a large share of those minutes. Run the heavy image build locally first.

## What can charge, if a later step creates it

| Resource | Why it appears | Order of magnitude | Free-plan note |
|---|---|---|---|
| NAT gateway | Private subnet outbound | About $32 per month plus $0.045 per GB | Do not create. Use a public task IP or App Runner |
| GPU instance | Inference or training | Dollars per hour | Do not create. CPU only |
| EC2 | A virtual machine | A `t3.micro` can fit a 12-month allowance on eligible paid accounts, then about $8 per month | Not part of this plan |
| RDS PostgreSQL | Managed database | A `db.t4g.micro` can fit a 12-month allowance, then about $12–15 per month plus storage | Do not create until Step 7 is approved |
| ElastiCache Redis | Celery broker | About $12 per month for a small node | Use SQS or a local Redis container instead |
| App Runner, 1 vCPU and 2 GB, always on | API | About $25–50 per month | Confirm the console allows it. Scale to the smallest CPU |
| Fargate, 0.5 vCPU and 1 GB, always on | API | About $15 per month | Same rule. One task |
| ECR | Image storage | About $0.10 per GB-month | A PyTorch image of 4–8 GB is about $0.40–0.80 per month if old tags are deleted |
| S3 | PDFs and artifacts | About $0.023 per GB-month | Invoice PDFs are small. The 0.5 GB checkpoint stored once is cents |
| CloudWatch logs | API logs | About $0.50 per GB ingested | Set retention to 7 days |
| Data transfer out | Downloads to browsers | Often an allowance, then about $0.09 per GB | PDFs are small. Do not serve the checkpoint from the bucket to the public |
| SQS | Job queue | 1 million requests per month on the always-free allowance | Preferred queue when AWS is used |

A forgotten NAT gateway or an always-on GPU is the expensive mistake. A forgotten ECR tag is the small one. A public S3 bucket is the security mistake, not the cost mistake.

## Image size is the main hidden cost

`model.pt` is about 501 MB and is gitignored. The Docker build context must not copy `artifacts/`, `data/docile/`, `kaggle_upload/`, virtual environments, or `node_modules`. The runtime image still needs CPU PyTorch, Transformers, and docTR. Expect several gigabytes. Store one tag in ECR and delete untagged images.

Do not download the checkpoint from the internet on every container start. Mount it or copy it once into a private bucket and pull it during deploy.

## What this project will not treat as free

- App Runner or Fargate running all month.
- RDS outside an accepted free-tier window, and any RDS storage beyond the allowance.
- NAT gateways.
- SageMaker training or hosting.
- A dedicated MLflow server on EC2.

MLflow on the laptop is the tracking tool until a cloud tracker has its own cost note.
