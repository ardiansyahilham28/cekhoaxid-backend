"""CekHoax ID — FastAPI backend.

Endpoints:
  GET  /health   — status service dan apakah model sudah termuat
  POST /predict  — klasifikasi teks: valid / hoaks (dengan flag tidak_yakin)
  POST /similar  — artikel referensi paling mirip (embedding + cosine similarity)

Model: IndoBERT hasil fine-tuning, tersimpan di ./models/indobert
(salinan dari MyDrive/CekHoaxID/models_v4/indobert di Google Drive —
model v4, retrain 24 Nov 2026, uji WA 18/20; v3 dan v2 tetap tersimpan
aman di Drive sebagai cadangan).

CATATAN PENTING — config model v1:
  config.json v1 tersimpan dengan _num_labels=5 padahal classifier head-nya
  2 label (classifier.weight berukuran [2, 768]). Model SELALU dimuat dengan
  num_labels=2 (lihat _load_model). Model hasil retrain (v2) menyimpan config
  yang benar, tapi pemuatan eksplisit ini tetap dipertahankan.

Ambang ketidakpastian (FR-05): probabilitas tertinggi < 0.60 -> tidak_yakin.
"""

import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.environ.get("CEKHOAX_MODEL_PATH", str(BASE_DIR / "models" / "indobert"))
CORPUS_PATH = os.environ.get("CEKHOAX_CORPUS_PATH", str(BASE_DIR / "reference_corpus.jsonl"))
EMBEDDING_MODEL_NAME = os.environ.get("CEKHOAX_EMBEDDING_MODEL", "")
UNCERTAINTY_THRESHOLD = float(os.environ.get("CEKHOAX_UNCERTAINTY_THRESHOLD", "0.60"))
MAX_LEN = 256
LABELS = {0: "valid", 1: "hoaks"}

app = FastAPI(title="CekHoax ID API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state = {"model": None, "tokenizer": None, "device": None, "model_error": None}
_corpus = {"items": None, "embeddings": None, "embedder": None}


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Teks yang dicek (mis. pesan WhatsApp)")


class PredictResponse(BaseModel):
    text: str
    label: int
    label_text: str
    prob_valid: float
    prob_hoaks: float
    confidence: float
    uncertain: bool
    model: str = "indobert-cekhoaxid"


class SimilarRequest(BaseModel):
    text: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)


class SimilarItem(BaseModel):
    text: str
    label: int
    label_text: str
    source: Optional[str] = None
    score: float


class SimilarResponse(BaseModel):
    query: str
    results: List[SimilarItem]


def _load_model():
    if _state["model"] is not None or _state["model_error"] is not None:
        return
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        # num_labels=2 eksplisit: config v1 salah menyimpan 5.
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)
        model.to(device)
        model.eval()
        _state.update(model=model, tokenizer=tokenizer, device=device)
    except Exception as e:  # noqa: BLE001 - sampaikan sebagai status, bukan crash
        _state["model_error"] = str(e)


def _require_model():
    _load_model()
    if _state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model belum termuat ({MODEL_PATH}): {_state['model_error']}. "
            "Salin folder model v4 dari Drive (MyDrive/CekHoaxID/models_v4/indobert) ke ./models/indobert "
            "atau atur env CEKHOAX_MODEL_PATH.",
        )
    return _state["model"], _state["tokenizer"], _state["device"]


@app.on_event("startup")
def _startup():
    _load_model()


@app.get("/health")
def health():
    _load_model()
    return {
        "status": "ok",
        "model_loaded": _state["model"] is not None,
        "model_path": MODEL_PATH,
        "model_error": _state["model_error"],
        "uncertainty_threshold": UNCERTAINTY_THRESHOLD,
    }


@app.get("/")
def root():
    return {"service": "CekHoax ID API", "docs": "/docs", "health": "/health"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model, tokenizer, device = _require_model()
    inputs = tokenizer(req.text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    label = int(np.argmax(probs))
    confidence = float(probs[label])
    return PredictResponse(
        text=req.text,
        label=label,
        label_text=LABELS[label],
        prob_valid=float(probs[0]),
        prob_hoaks=float(probs[1]),
        confidence=confidence,
        uncertain=confidence < UNCERTAINTY_THRESHOLD,
    )


def _mean_pool_embedding(texts):
    """Embedding dari encoder IndoBERT (mean pooling). Dipakai bila tidak ada
    sentence-transformers / CEKHOAX_EMBEDDING_MODEL."""
    model, tokenizer, device = _require_model()
    encoder = model.base_model
    inputs = tokenizer(
        texts, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=True
    ).to(device)
    with torch.no_grad():
        hidden = encoder(**inputs).last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    emb = summed / counts
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()


def _embed(texts):
    if EMBEDDING_MODEL_NAME:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise HTTPException(
                status_code=503, detail=f"sentence-transformers tidak terpasang: {e}"
            )
        if _corpus["embedder"] is None:
            _corpus["embedder"] = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return np.asarray(
            _corpus["embedder"].encode(texts, normalize_embeddings=True)
        )
    return _mean_pool_embedding(texts)


def _load_corpus():
    if _corpus["items"] is not None:
        return _corpus["items"]
    path = Path(CORPUS_PATH)
    if not path.exists():
        raise HTTPException(
            status_code=503, detail=f"Reference corpus tidak ditemukan: {CORPUS_PATH}"
        )
    items = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not items:
        raise HTTPException(status_code=503, detail="Reference corpus kosong.")
    _corpus["items"] = items
    cache = path.with_suffix(".embeddings.npy")
    if cache.exists():
        try:
            _corpus["embeddings"] = np.load(cache)
        except Exception:
            _corpus["embeddings"] = None
    if _corpus["embeddings"] is None or len(_corpus["embeddings"]) != len(items):
        embs = _embed([it["text"] for it in items])
        _corpus["embeddings"] = embs
        np.save(cache, embs)
    return items


@app.post("/similar", response_model=SimilarResponse)
def similar(req: SimilarRequest):
    items = _load_corpus()
    embs = _corpus["embeddings"]
    query_emb = _embed([req.text])[0]
    scores = embs @ query_emb
    top_idx = np.argsort(-scores)[: req.top_k]
    results = [
        SimilarItem(
            text=items[i]["text"],
            label=int(items[i]["label"]),
            label_text=LABELS[int(items[i]["label"])],
            source=items[i].get("source"),
            score=float(scores[i]),
        )
        for i in top_idx
    ]
    return SimilarResponse(query=req.text, results=results)
