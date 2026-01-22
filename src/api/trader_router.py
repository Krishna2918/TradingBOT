"""
Trader Router

FastAPI service that routes short prompts to a local Ollama model (RTX 4080)
and long/deep prompts to Ollama Cloud (gpt-oss:120b-cloud). Includes
health/metrics, logging, and a kill switch.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Dict, Any

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# --- Logging setup ---
import json as json_lib

LOG_DIR = os.environ.get("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# JSON formatter for structured logs
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Add extra fields if present
        if hasattr(record, "engine_used"):
            log_data["engine_used"] = record.engine_used
        if hasattr(record, "latency_ms"):
            log_data["latency_ms"] = record.latency_ms
        if hasattr(record, "chars"):
            log_data["chars"] = record.chars
        if hasattr(record, "route_decision"):
            log_data["route_decision"] = record.route_decision
        return json_lib.dumps(log_data)

json_formatter = JSONFormatter()

# File handler with JSON
json_file_handler = logging.FileHandler(
    os.path.join(LOG_DIR, "app.json.log"), encoding="utf-8"
)
json_file_handler.setFormatter(json_formatter)

# Console handler (human-readable)
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
)

# Configure root logger
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    handlers=[json_file_handler, console_handler],
)

logger = logging.getLogger("trader_router")


# --- Configuration ---
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "llama3:8b")
CLOUD_MODEL = os.environ.get("CLOUD_MODEL", "gpt-oss:120b-cloud")
EMBED_SERVICE_URL = os.environ.get("EMBED_SERVICE_URL", "http://127.0.0.1:8011")
ROUTE_THRESHOLD = int(os.environ.get("ROUTE_THRESHOLD", "600"))


class DecideRequest(BaseModel):
    prompt: str


class DecideResponse(BaseModel):
    engine: str
    reply: str
    latency_ms: int


class EmbedRequest(BaseModel):
    text: str


app = FastAPI(title="AI Trader Router", version="1.0.0")


def _ollama_generate(model: str, prompt: str, timeout: int = 60) -> str:
    """Call Ollama generate HTTP API for a given model."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama request failed for {model}: {e}")


def _kill_switch_active() -> bool:
    return os.environ.get("KILL_SWITCH", "0") == "1"


@app.get("/health")
def health() -> Dict[str, Any]:
    status = {
        "service": "ok",
        "kill_switch": _kill_switch_active(),
        "models": {
            "local": LOCAL_MODEL,
            "cloud": CLOUD_MODEL,
        },
        "ollama_url": OLLAMA_URL,
    }
    # Try quick tag fetch to verify Ollama service
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        status["ollama_available"] = resp.status_code == 200
    except Exception:
        status["ollama_available"] = False
    return status


@app.post("/decide", response_model=DecideResponse)
def decide(req: DecideRequest) -> DecideResponse:
    if _kill_switch_active():
        logger.critical("KILL_SWITCH active; blocking /decide")
        raise HTTPException(status_code=503, detail="Kill switch active; trading decisions blocked.")

    prompt = req.prompt
    prompt_len = len(prompt)
    route_decision = "local" if prompt_len < ROUTE_THRESHOLD else "cloud"
    target_model = LOCAL_MODEL if route_decision == "local" else CLOUD_MODEL

    t0 = time.perf_counter()
    try:
        reply = _ollama_generate(target_model, prompt)
    except Exception as e:
        logger.warning(f"Primary engine {target_model} failed: {e}")
        # Fallback: if cloud failed, try local; if local failed, try cloud
        fallback = LOCAL_MODEL if target_model == CLOUD_MODEL else CLOUD_MODEL
        try:
            reply = _ollama_generate(fallback, prompt)
            target_model = fallback
            route_decision = "fallback_to_" + ("local" if fallback == LOCAL_MODEL else "cloud")
        except Exception as e2:
            logger.error(f"Fallback engine also failed: {e2}")
            raise HTTPException(status_code=502, detail=f"Inference failed: {e2}")

    latency_ms = int((time.perf_counter() - t0) * 1000)

    logger.info(
        "inference_completed",
        extra={
            "engine_used": target_model,
            "latency_ms": latency_ms,
            "chars": prompt_len,
            "route_decision": route_decision,
        },
    )

    return DecideResponse(engine=target_model, reply=reply, latency_ms=latency_ms)


@app.post("/embed")
def embed(req: EmbedRequest) -> Dict[str, Any]:
    try:
        r = requests.post(f"{EMBED_SERVICE_URL}/embed", json={"text": req.text}, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding service error: {e}")


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    # Minimal metrics snapshot; extend as needed
    return {
        "local_model": LOCAL_MODEL,
        "cloud_model": CLOUD_MODEL,
        "route_threshold": ROUTE_THRESHOLD,
        "kill_switch": _kill_switch_active(),
    }

