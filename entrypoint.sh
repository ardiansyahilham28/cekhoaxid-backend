#!/bin/sh
set -e
# Unduh model v4 dari Hugging Face Hub (repo publik, tanpa token) ke folder lokal backend.
python -c "from huggingface_hub import snapshot_download; snapshot_download('hamsyah/cekhoaxid-v4', local_dir='models/indobert')"
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-7860}"
