"""
Embeddings Service (FastAPI)

Serves sentence embeddings at http://127.0.0.1:8011 via sentence-transformers.
Defaults to 'sentence-transformers/all-MiniLM-L6-v2'. If OpenVINO is configured
on the system, transformers may leverage it automatically; otherwise CPU runs.
"""

from __future__ import annotations

import os
import logging
from typing import List, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer


LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s [emb] %(message)s")
logger = logging.getLogger("embeddings_service")

EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

app = FastAPI(title="Embeddings Service", version="1.0.0")
_model: SentenceTransformer | None = None
_model_loading: bool = False


class EmbedOneRequest(BaseModel):
    text: str


class EmbedBatchRequest(BaseModel):
    texts: List[str]


def _ensure_model_loaded() -> SentenceTransformer:
    """Lazy-load model on first request instead of startup."""
    global _model, _model_loading
    
    if _model is not None:
        return _model
    
    if _model_loading:
        # Model is currently being loaded by another request
        import time
        max_wait = 60  # 60 seconds max wait
        waited = 0
        while _model_loading and waited < max_wait:
            time.sleep(0.5)
            waited += 0.5
        if _model is None:
            raise RuntimeError("Model loading timed out")
        return _model
    
    # Load model
    _model_loading = True
    try:
        logger.info(f"Lazy-loading embeddings model: {EMBED_MODEL}")
        _model = SentenceTransformer(EMBED_MODEL)
        logger.info(f"Model loaded successfully: {EMBED_MODEL}")
        return _model
    finally:
        _model_loading = False


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model": EMBED_MODEL,
        "loaded": _model is not None,
        "loading": _model_loading
    }


@app.post("/embed")
def embed(req: EmbedOneRequest) -> Dict[str, Any]:
    model = _ensure_model_loaded()
    vec = model.encode(req.text, normalize_embeddings=True).tolist()
    return {"embedding": vec, "dim": len(vec), "model": EMBED_MODEL}


@app.post("/embed_batch")
def embed_batch(req: EmbedBatchRequest) -> Dict[str, Any]:
    model = _ensure_model_loaded()
    vecs = model.encode(req.texts, normalize_embeddings=True)
    return {
        "embeddings": [v.tolist() for v in vecs],
        "dim": len(vecs[0]) if len(vecs) else 0,
        "count": len(vecs),
        "model": EMBED_MODEL,
    }

