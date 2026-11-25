"""CekHoax ID — FastAPI backend.

Live backend: http://103.42.245.54:7860/ (VPS, systemd service `cekhoaxid`).
Repo ini (main.py) adalah implementasi referensi sekaligus sumber deploy
(git pull di VPS lalu restart service).

Endpoints (kontrak live, Bab 11):
  GET  /health   — status service dan apakah model sudah termuat
  POST /predict  — klasifikasi teks: valid / hoaks / tidak_yakin
                   (teks 20-5000 karakter; batas 30 req/menit per IP;
                   mengembalikan kata_penting)
  POST /predict-url — klasifikasi isi artikel dari URL
  POST /similar  — artikel referensi paling mirip (dengan url_klarifikasi)
  GET  /berita?page=&limit=   — daftar berita terbaru (Antara + Google News ID)
  GET  /cek-fakta?page=&limit= — daftar klarifikasi TurnBackHoax
  POST /feedback — simpan umpan balik klasifikasi

Model: IndoBERT hasil fine-tuning, model v4 (uji WA 18/20).
Bila folder model tidak ada, model diunduh otomatis dari Hugging Face Hub
(repo publik hamsyah/cekhoaxid-v4) saat start.

CATATAN PENTING — config model v1:
  config.json v1 tersimpan dengan _num_labels=5 padahal classifier head-nya
  2 label (classifier.weight berukuran [2, 768]). Model SELALU dimuat dengan
  num_labels=2 (lihat _load_model). Model hasil retrain (v2) menyimpan config
  yang benar, tapi pemuatan eksplisit ini tetap dipertahankan.

Aturan label (Bab 11): prob_hoaks >= 0.60 -> hoaks; prob_valid >= 0.60 ->
valid; selain itu -> tidak_yakin.
"""

import html as html_lib
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from collections import deque
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import List, Optional

import httpx
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.environ.get("CEKHOAX_MODEL_PATH", str(BASE_DIR / "models" / "indobert"))
CORPUS_PATH = os.environ.get("CEKHOAX_CORPUS_PATH", str(BASE_DIR / "claims_corpus.jsonl"))
EMBEDDING_MODEL_NAME = os.environ.get("CEKHOAX_EMBEDDING_MODEL", "")
UNCERTAINTY_THRESHOLD = float(os.environ.get("CEKHOAX_UNCERTAINTY_THRESHOLD", "0.60"))
MAX_LEN = 256
LABELS = {0: "valid", 1: "hoaks"}
TEXT_MIN_LEN = 20
TEXT_MAX_LEN = 5000
RATE_LIMIT_WINDOW_S = 60
RATE_LIMIT_MAX = 30
FEED_CACHE_TTL_S = 1800
FEED_TIMEOUT_S = 15
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

app = FastAPI(title="CekHoax ID API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state = {"model": None, "tokenizer": None, "device": None, "model_error": None}
_corpus = {"items": None, "embeddings": None, "embedder": None}
_rate_limit_hits = {}
_feed_cache = {}


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=TEXT_MIN_LEN,
        max_length=TEXT_MAX_LEN,
        description="Teks yang dicek (mis. pesan WhatsApp), 20-5000 karakter",
    )


class PredictResponse(BaseModel):
    text: str
    label: int
    label_text: str
    prob_valid: float
    prob_hoaks: float
    confidence: float
    uncertain: bool
    kata_penting: List[str] = []
    model: str = "indobert-cekhoaxid"


class PredictUrlRequest(BaseModel):
    url: str = Field(..., min_length=1, description="URL artikel http/https")


class PredictUrlResponse(BaseModel):
    judul: str
    label_text: str
    confidence: float
    kata_penting: List[str] = []


class SimilarRequest(BaseModel):
    text: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    min_score: float = Field(0.70, ge=0.0, le=1.0)


class SimilarItem(BaseModel):
    text: str
    label: int
    label_text: str
    source: Optional[str] = None
    url_klarifikasi: Optional[str] = None
    score: float


class SimilarResponse(BaseModel):
    query: str
    results: List[SimilarItem]


class BeritaItem(BaseModel):
    id: str
    judul: str
    sumber: str
    tanggal: Optional[str] = None
    url: Optional[str] = None
    ringkasan: Optional[str] = None
    label: str
    confidence: float


class CekFaktaItem(BaseModel):
    id: str
    judul: str
    sumber: str
    tanggal: Optional[str] = None
    url: Optional[str] = None
    ringkasan: Optional[str] = None


class FeedbackRequest(BaseModel):
    text: str = Field(..., min_length=1)
    prediksi: str = Field(..., min_length=1)
    label_benar: str = Field(..., min_length=1)
    catatan: Optional[str] = None


def _load_model():
    if _state["model"] is not None or _state["model_error"] is not None:
        return
    try:
        model_dir = Path(MODEL_PATH)
        if not (model_dir / "config.json").exists():
            # Deploy dari GitHub (mis. Altare) tidak menjalankan
            # entrypoint.sh, jadi model v4 diunduh dari Hugging Face Hub
            # (repo publik hamsyah/cekhoaxid-v4, tanpa token) saat start.
            print(
                f"Model tidak ditemukan di {MODEL_PATH}; "
                "mengunduh dari hamsyah/cekhoaxid-v4 ...",
                flush=True,
            )
            from huggingface_hub import snapshot_download

            snapshot_download("hamsyah/cekhoaxid-v4", local_dir=str(model_dir))
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
            "Pastikan model v4 tersedia di folder itu, atau biarkan backend "
            "mengunduhnya otomatis dari hamsyah/cekhoaxid-v4 di Hugging Face Hub.",
        )
    return _state["model"], _state["tokenizer"], _state["device"]


def _label_from_probs(probs):
    """Aturan label Bab 11: >= 0.60 -> kelas itu, selain itu tidak_yakin."""
    p_valid = float(probs[0])
    p_hoaks = float(probs[1])
    if p_hoaks >= UNCERTAINTY_THRESHOLD:
        return 1, "hoaks", p_hoaks
    if p_valid >= UNCERTAINTY_THRESHOLD:
        return 0, "valid", p_valid
    label = int(np.argmax(probs))
    return label, "tidak_yakin", float(probs[label])


def _check_rate_limit(request: Request):
    """Batas 30 request/menit per IP untuk /predict (429 bila lewat)."""
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    hits = _rate_limit_hits.setdefault(ip, deque())
    while hits and hits[0] <= now - RATE_LIMIT_WINDOW_S:
        hits.popleft()
    if len(hits) >= RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429,
            detail="Terlalu banyak permintaan. Coba lagi sebentar lagi.",
            headers={"Retry-After": str(RATE_LIMIT_WINDOW_S)},
        )
    hits.append(now)


def _important_words(text: str, top_k: int = 5) -> List[str]:
    """Kata paling berpengaruh terhadap prediksi (saliensi gradien).

    Satu forward + backward: gradien logit kelas terprediksi terhadap
    embedding kata, dikali embedding (gradient x input), diagregasi per kata.
    """
    model, tokenizer, device = _require_model()
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    embeds = model.get_input_embeddings()(input_ids)
    embeds.retain_grad()
    outputs = model(inputs_embeds=embeds, attention_mask=attn)
    logits = outputs.logits
    pred = int(torch.argmax(logits, dim=-1))
    model.zero_grad()
    logits[0, pred].backward()
    saliency = (embeds.grad * embeds).sum(dim=-1).abs()[0].detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    words = []  # (kata, skor)
    for tok, score in zip(tokens, saliency):
        if tok in ("[CLS]", "[SEP]", "[PAD]", "[MASK]"):
            continue
        if tok.startswith("##"):
            if words:
                prev, prev_score = words[-1]
                words[-1] = (prev + tok[2:], prev_score + float(score))
            continue
        words.append((tok, float(score)))
    ranked = [
        (w, s)
        for w, s in sorted(words, key=lambda x: x[1], reverse=True)
        if any(ch.isalnum() for ch in w)
    ]
    return [w for w, _ in ranked[:top_k]]


def _classify_texts(texts: List[str]):
    """Klasifikasi batch (satu forward pass). Kembalikan list
    (label_text, confidence) per teks."""
    model, tokenizer, device = _require_model()
    enc = tokenizer(
        texts, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=True
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return [_label_from_probs(p)[1:] for p in probs]


def _http_client(timeout_s: float) -> httpx.Client:
    """Client HTTP yang memakai proxy egress dari env bila ada (mesin Muse),
    dan koneksi langsung bila tidak ada (VPS). trust_env=False agar httpx
    tidak gagal mem-parse NO_PROXY berisi alamat IPv6."""
    proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    # Pakai bundle CA sistem (berisi CA proxy egress di mesin Muse); bila
    # tidak ada, pakai default httpx/certifi.
    verify: object = True
    if Path("/etc/ssl/certs/ca-certificates.crt").exists():
        verify = "/etc/ssl/certs/ca-certificates.crt"
    return httpx.Client(
        follow_redirects=True,
        timeout=timeout_s,
        headers={"User-Agent": BROWSER_UA},
        proxy=proxy,
        trust_env=False,
        verify=verify,
    )


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
def predict(req: PredictRequest, request: Request):
    _check_rate_limit(request)
    model, tokenizer, device = _require_model()
    inputs = tokenizer(req.text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    label, label_text, confidence = _label_from_probs(probs)
    return PredictResponse(
        text=req.text,
        label=label,
        label_text=label_text,
        prob_valid=float(probs[0]),
        prob_hoaks=float(probs[1]),
        confidence=confidence,
        uncertain=label_text == "tidak_yakin",
        kata_penting=_important_words(req.text),
    )


@app.post("/predict-url", response_model=PredictUrlResponse)
def predict_url(req: PredictUrlRequest):
    parsed = httpx.URL(req.url)
    if parsed.scheme not in ("http", "https") or not parsed.host:
        raise HTTPException(
            status_code=422, detail="URL harus memakai http/https dan dapat dijangkau."
        )
    try:
        with _http_client(20) as client:
            resp = client.get(req.url)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=422, detail=f"URL tidak dapat dijangkau: {e}")
    if resp.status_code != 200:
        raise HTTPException(
            status_code=422,
            detail=f"URL tidak dapat dijangkau (HTTP {resp.status_code}).",
        )
    try:
        from trafilatura import extract, extract_metadata
    except ImportError as e:
        raise HTTPException(
            status_code=503, detail=f"Ekstraktor artikel tidak terpasang: {e}"
        )
    article_text = extract(
        resp.text, include_comments=False, include_tables=False
    ) or ""
    metadata = extract_metadata(resp.text)
    judul = ""
    if metadata is not None:
        judul = metadata.title or ""
    if not judul:
        m = re.search(r"<title[^>]*>(.*?)</title>", resp.text, re.I | re.S)
        judul = html_lib.unescape(re.sub(r"<[^>]+>", "", m.group(1)).strip()) if m else req.url
    text = article_text.strip()
    if len(text) < TEXT_MIN_LEN:
        raise HTTPException(
            status_code=422,
            detail="Isi artikel tidak bisa diekstrak dari URL ini.",
        )
    model, tokenizer, device = _require_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    _, label_text, confidence = _label_from_probs(probs)
    return PredictUrlResponse(
        judul=judul,
        label_text=label_text,
        confidence=confidence,
        kata_penting=_important_words(text),
    )


def _mean_pool_embedding(texts):
    """Embedding dari encoder IndoBERT (mean pooling). Dipakai bila tidak ada
    sentence-transformers / CEKHOAX_EMBEDDING_MODEL. Diproses per batch 32
    agar korpus 1000 klaim tidak meledak di memori CPU."""
    model, tokenizer, device = _require_model()
    encoder = model.base_model
    out = []
    for start in range(0, len(texts), 32):
        batch = texts[start : start + 32]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=True
        ).to(device)
        with torch.no_grad():
            hidden = encoder(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        emb = summed / counts
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        out.append(emb.cpu().numpy())
    return np.concatenate(out, axis=0)


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
    # Hanya hasil di atas ambang yang dikembalikan; bila kurang dari 2 hasil,
    # kembalikan list kosong supaya app menyembunyikan section "klaim mirip"
    # (satu hasil di atas ambang saja tidak cukup informatif).
    ranked = np.argsort(-scores)
    picked = [i for i in ranked if scores[i] >= req.min_score][: req.top_k]
    if len(picked) < 2:
        return SimilarResponse(query=req.text, results=[])
    results = [
        SimilarItem(
            text=items[i]["text"],
            label=int(items[i]["label"]),
            label_text=LABELS[int(items[i]["label"])],
            source=items[i].get("source"),
            url_klarifikasi=items[i].get("url") or items[i].get("url_klarifikasi"),
            score=float(scores[i]),
        )
        for i in picked
    ]
    return SimilarResponse(query=req.text, results=results)


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return html_lib.unescape(re.sub(r"\s+", " ", text)).strip()


def _parse_date(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw.strip()).isoformat()
    except (ValueError, TypeError):
        pass
    try:
        return datetime.fromisoformat(raw.strip().replace("Z", "+00:00")).isoformat()
    except (ValueError, TypeError):
        return None


def _fetch_feed(url: str, source_name: str):
    """Ambil satu feed RSS/Atom. Hasil di-cache 30 menit; bila gagal dan ada
    cache lama, pakai cache lama; bila tidak ada, kembalikan list kosong."""
    now = time.monotonic()
    cached = _feed_cache.get(url)
    if cached and now - cached[0] < FEED_CACHE_TTL_S:
        return cached[1]
    items = []
    try:
        with _http_client(FEED_TIMEOUT_S) as client:
            resp = client.get(url)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        entries = root.findall(".//item")
        if not entries:
            entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")
        for entry in entries:
            title = entry.findtext("title") or ""
            link = entry.findtext("link") or ""
            if not link:
                link_el = entry.find("{http://www.w3.org/2005/Atom}link")
                if link_el is not None:
                    link = link_el.get("href") or ""
            desc = (
                entry.findtext("description")
                or entry.findtext("{http://purl.org/rss/1.0/modules/content/}encoded")
                or entry.findtext("{http://www.w3.org/2005/Atom}summary")
                or ""
            )
            ringkasan = _strip_html(desc)[:300]
            tanggal = _parse_date(
                entry.findtext("pubDate")
                or entry.findtext("{http://purl.org/dc/elements/1.1/}date")
                or entry.findtext("{http://www.w3.org/2005/Atom}updated")
            )
            guid = entry.findtext("guid") or link
            judul = title.strip()
            sumber = source_name
            if source_name == "Google News" and " - " in judul:
                judul, _, sumber = judul.rpartition(" - ")
                judul = judul.strip()
                sumber = sumber.strip() or source_name
            if judul:
                items.append(
                    {
                        "id": guid.strip() if guid else link,
                        "judul": judul,
                        "sumber": sumber,
                        "tanggal": tanggal,
                        "url": link.strip() or None,
                        "ringkasan": ringkasan or None,
                    }
                )
    except Exception as e:  # noqa: BLE001 - feed gagal: pakai cache/daftar kosong
        print(f"Gagal mengambil feed {url}: {e}", flush=True)
        if cached:
            return cached[1]
        return []
    _feed_cache[url] = (now, items)
    return items


def _paginate(items: list, page: int, limit: int) -> list:
    page = max(1, page)
    limit = min(max(1, limit), 50)
    start = (page - 1) * limit
    return items[start : start + limit]


@app.get("/berita", response_model=List[BeritaItem])
def berita(page: int = 1, limit: int = 10):
    """Daftar berita terbaru (Antara + Google News Indonesia), terklasifikasi
    model. Diurutkan terbaru dulu."""
    raw = _fetch_feed("https://www.antaranews.com/rss/terkini.xml", "Antara") + (
        _fetch_feed(
            "https://news.google.com/rss?hl=id&gl=ID&ceid=ID:id", "Google News"
        )
    )
    raw.sort(key=lambda it: it.get("tanggal") or "", reverse=True)
    fresh = raw[:30]
    if fresh:
        try:
            labels = _classify_texts(
                [(it["judul"] + " " + (it["ringkasan"] or ""))[:2000] for it in fresh]
            )
        except HTTPException:
            labels = [("tidak_yakin", 0.0)] * len(fresh)
        classified = {}
        for it, (label_text, conf) in zip(fresh, labels):
            classified[it["id"]] = (label_text, conf)
    else:
        classified = {}
    items = [
        BeritaItem(
            id=it["id"],
            judul=it["judul"],
            sumber=it["sumber"],
            tanggal=it["tanggal"],
            url=it["url"],
            ringkasan=it["ringkasan"],
            label=classified.get(it["id"], ("tidak_yakin", 0.0))[0],
            confidence=classified.get(it["id"], ("tidak_yakin", 0.0))[1],
        )
        for it in raw
    ]
    return _paginate(items, page, limit)


@app.get("/cek-fakta", response_model=List[CekFaktaItem])
def cek_fakta(page: int = 1, limit: int = 10):
    """Daftar klarifikasi terbaru dari TurnBackHoax, terbaru dulu."""
    raw = _fetch_feed("https://turnbackhoax.id/feed", "TurnBackHoax")
    raw.sort(key=lambda it: it.get("tanggal") or "", reverse=True)
    items = [
        CekFaktaItem(
            id=it["id"],
            judul=it["judul"],
            sumber=it["sumber"],
            tanggal=it["tanggal"],
            url=it["url"],
            ringkasan=it["ringkasan"],
        )
        for it in raw
    ]
    return _paginate(items, page, limit)


@app.post("/feedback")
def feedback(req: FeedbackRequest, request: Request):
    """Simpan umpan balik klasifikasi (append ke feedback.jsonl)."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "ip": request.client.host if request.client else None,
        "text": req.text,
        "prediksi": req.prediksi,
        "label_benar": req.label_benar,
        "catatan": req.catatan,
    }
    with open(BASE_DIR / "feedback.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"status": "ok"}
