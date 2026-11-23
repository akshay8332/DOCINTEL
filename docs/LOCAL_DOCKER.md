# Local Docker

This runs the invoice app on the laptop. It does not create AWS resources. The image does not contain epoch-3. Compose mounts `artifacts/checkpoints/epoch-3` read-only.

The selected model stays epoch-3, threshold 0.50, grouping `line-block-v2`, and correction stack `stack-v1:correct+boundary+fallback`.

## What the containers do

- `api` is the FastAPI process. It installs CPU PyTorch and docTR. `DOCINTEL_DOCTR_PYTHON` points at that interpreter, so the Windows `.venv-doctr` path is not used inside the container.
- `web` is nginx serving `apps/web/dist`. Run `npm run build` in `apps/web` before Compose if that folder is missing. Nginx forwards `/api` to the API, so the browser stays on one origin.
- SQLite and uploads live in the Compose volume `app-data`. That volume is separate from `data/app` and `data/uploads` used by the virtual environment.
- `docker/Dockerfile.worker` is the old training sketch. Compose does not start it.

## Start

From the repository root:

```powershell
docker compose up --build
```

Open `http://localhost:5173`. The API is also on `http://localhost:8080/health`. MLflow is on `http://localhost:5000`.

Register the frozen checkpoint once, after MLflow is healthy:

```powershell
docker compose exec mlflow python /register.py
```

The model name is `docile-address-epoch3`, alias `benchmark`. The script does not train.

The first build downloads PyTorch and docTR and takes a long time. Later builds reuse layers.

## Health

`/health` reports `inference_mode` `real` when `artifacts/checkpoints/epoch-3/model.pt` is visible in the container. If that file is missing, the mode is `unavailable` and a new upload is not a model prediction.

## Stop

```powershell
docker compose down
```

`docker compose down -v` also deletes the Compose database and uploaded PDFs. It does not delete the mounted checkpoint.
