# DocIntel

Local-first **document intelligence** pipeline: FUNSD → LayoutLM fine-tune → extraction API → Vue monitor.

The invoice application (DOCILE) is documented in [docs/PROJECT_ASSESSMENT.md](docs/PROJECT_ASSESSMENT.md) and [docs/LOCAL_STARTUP.md](docs/LOCAL_STARTUP.md). Start the API with `uvicorn docintel.api.main:app` and the UI with `npm run dev` inside `apps/web`.

Stage 1 (this scaffold) focuses on **FUNSD download/parse**, **LayoutLMv3** training (v2 fallback), and entity F1 metrics. Later stages are stubbed under `services/`, `api/`, and `apps/monitor`.

## Requirements

- Windows 10/11 (PowerShell), Python **3.10+**
- Optional: NVIDIA GPU + CUDA-enabled PyTorch for faster training
- Modest GPU defaults: batch size `2`, gradient accumulation `4`

## Setup (Windows)

```powershell
cd f:\Projects\docintel
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev]"
```

CPU-only PyTorch is fine for a smoke train. For CUDA, install the matching wheel from [pytorch.org](https://pytorch.org) **before** or after editable install if needed.

Copy env template:

```powershell
Copy-Item .env.example .env
```

## Download FUNSD

```powershell
python scripts\download_funsd.py --dest data\funsd
```

## Train (full / smoke)

Config: `configs\training.yaml` (Pydantic-loaded).

**Smoke (1–2 steps, cheap local check):**

```powershell
python -m docintel.workers.train_funsd --smoke
```

**Fuller local train:**

```powershell
python -m docintel.workers.train_funsd --config configs\training.yaml
```

Useful flags: `--batch-size 2 --grad-accum 4 --max-steps 50 --device auto`

Checkpoints + metrics land under `artifacts\checkpoints\funsd-layoutlm\`.

## Eval a checkpoint

```powershell
python scripts\eval_checkpoint.py --checkpoint artifacts\checkpoints\funsd-layoutlm
```

## Tests

```powershell
pytest tests\unit -q
```

## Layout

See the phased plan — package tree under `packages\docintel\` (schemas, data, models, services, api, workers), Vue placeholder in `apps\monitor`, compose/Docker sketches in `infra\` / `docker\`.

## Design rules

- **Schemas / config:** Pydantic v2 only
- **Services:** no FastAPI imports
- **Workers & API:** same model / extract code path

## License note

FUNSD and Hugging Face model cards have their own terms — verify before commercial use.
