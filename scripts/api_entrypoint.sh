#!/bin/sh
set -eu
python /app/scripts/fetch_checkpoint.py
exec uvicorn docintel.api.main:app --host 0.0.0.0 --port 8080
