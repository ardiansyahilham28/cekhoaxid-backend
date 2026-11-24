# CekHoax ID — Backend API

FastAPI backend untuk klasifikasi hoaks berbahasa Indonesia (IndoBERT fine-tuned, model v4).

- `GET /health` — status service dan apakah model sudah termuat
- `POST /predict` — klasifikasi teks: `valid` / `hoaks` (dengan flag `uncertain`)
- `POST /similar` — artikel referensi paling mirip

Model: [hamsyah/cekhoaxid-v4](https://huggingface.co/hamsyah/cekhoaxid-v4) — diunduh otomatis saat container dijalankan (repo publik, tanpa token). Uji WA 20 pesan: 18/20 (valid 8/10, hoaks 10/10).

## Menjalankan lokal

```bash
python3 -m venv .venv
.venv/bin/pip install "torch>=2.0" --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
```

## Deploy (Docker)

```bash
docker build -t cekhoaxid-api .
docker run -p 7860:7860 cekhoaxid-api
```

Port diambil dari env `PORT` (default 7860).
