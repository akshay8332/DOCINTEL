# Continuous integration

GitHub Actions runs from `.github/workflows/ci.yml`. It does not deploy to AWS and it does not train or replace epoch-3.

## Jobs

| Job | When | What |
|---|---|---|
| `test` | every push and PR | `pytest` with CPU PyTorch |
| `frontend` | every push and PR | `npm ci` and `npm run build` in `apps/web` |
| `docker-web` | after test + frontend | builds `docker/Dockerfile.web` |
| `docker-api` | `main` push or manual run | builds `docker/Dockerfile.api` |

None of these jobs push an image to ECR. That is a later, cost-checked step.

The workflow uses Actions that run on Node 24 (`checkout@v5`, `setup-python@v6`, `setup-node@v5`, `upload-artifact@v6`). The Node 20 deprecation warning on older action majors is expected until those majors are upgraded.

## Watching a run

On GitHub: **Actions** → **CI**. A green run means the tests and builds passed on a clean machine. The containers on this laptop are unchanged until you rebuild them locally.
