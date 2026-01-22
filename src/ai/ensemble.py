"""
Ollama ensemble coordinator for AI stock selection.

Three large language models evaluate the factor payload and vote on the final
set of symbols. Responses are validated against a strict JSON schema to avoid
hallucinated formats.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import zlib
from dataclasses import dataclass
from typing import Dict, List, Sequence

import requests
from jsonschema import Draft7Validator


logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_ENDPOINT = "/api/chat"

# Default models - can be overridden via OLLAMA_MODELS env var (comma-separated)
DEFAULT_MODELS = [
    "qwen3-coder:480b-cloud",
    "deepseek-v3.1:671b-cloud",
    "gpt-oss:120b",
]

# Fallback smaller models that are more likely to be available locally
FALLBACK_MODELS = [
    "llama3:8b",
    "mistral:7b",
    "qwen2:7b",
]

def _get_models() -> List[str]:
    """Get model list from environment or use defaults."""
    env_models = os.getenv("OLLAMA_MODELS")
    if env_models:
        return [m.strip() for m in env_models.split(",") if m.strip()]
    return DEFAULT_MODELS

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "choices": {
            "type": "array",
            "minItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "minLength": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "explanation": {"type": "string", "minLength": 10},
                },
                "required": ["symbol", "confidence", "explanation"],
            },
        }
    },
    "required": ["choices"],
}
VALIDATOR = Draft7Validator(RESPONSE_SCHEMA)


class EnsembleError(RuntimeError):
    """Raised when the ensemble cannot produce a valid response."""


@dataclass
class EnsembleSelection:
    symbol: str
    confidence: float
    explanation: str
    votes: int


class EnsembleCoordinator:
    """
    Coordinate model voting via the local Ollama endpoint.

    Falls back to score-based selection if no models are available.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_URL,
        models: Sequence[str] = None,
        request_timeout: int = 120,
        use_fallback_models: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.models = list(models) if models else _get_models()
        self.fallback_models = FALLBACK_MODELS if use_fallback_models else []
        self.request_timeout = request_timeout

    def run(
        self,
        feature_payload: Dict[str, dict],
        score_payload: Dict[str, dict],
    ) -> List[EnsembleSelection]:
        """
        Execute the ensemble and return five selected symbols.

        Falls back to score-based selection if all models fail.
        """
        # Check if Ollama is available
        if not self._check_ollama_available():
            logger.warning("Ollama not available, using score-based fallback.")
            return self._score_based_fallback(score_payload)

        compressed_payload = self._compress_payload(
            {"features": feature_payload, "scores": score_payload}
        )

        model_outputs = []
        all_models = self.models + self.fallback_models

        for model in all_models:
            try:
                model_outputs.append(self._invoke_model(model, compressed_payload))
                logger.info("Model %s responded successfully", model)
            except Exception as exc:
                logger.warning("Model %s failed: %s", model, exc)
                continue

        if not model_outputs:
            logger.warning("All ensemble models failed, using score-based fallback.")
            return self._score_based_fallback(score_payload)

        aggregated: Dict[str, Dict[str, List]] = {}
        for response in model_outputs:
            for choice in response["choices"]:
                symbol = choice["symbol"].upper()
                aggregated.setdefault(symbol, {"confidence": [], "explanation": []})
                aggregated[symbol]["confidence"].append(choice["confidence"])
                aggregated[symbol]["explanation"].append(choice["explanation"])

        if not aggregated:
            raise EnsembleError("No symbols returned by ensemble models.")

        selections = []
        for symbol, payload in aggregated.items():
            votes = len(payload["confidence"])
            mean_confidence = sum(payload["confidence"]) / votes
            explanation = payload["explanation"][0]
            selections.append(
                EnsembleSelection(
                    symbol=symbol,
                    confidence=mean_confidence,
                    explanation=explanation,
                    votes=votes,
                )
            )

        selections.sort(key=lambda s: (s.votes, s.confidence), reverse=True)
        top = selections[:5]

        # If fewer than five meet majority, pad with highest remaining confidences.
        if len(top) < 5:
            remaining = [s for s in selections if s.symbol not in {t.symbol for t in top}]
            remaining.sort(key=lambda s: s.confidence, reverse=True)
            for s in remaining:
                if len(top) >= 5:
                    break
                top.append(s)

        if len(top) < 5:
            raise EnsembleError("Ensemble produced fewer than five selections.")

        return top[:5]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _compress_payload(self, payload: Dict) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        compressed = zlib.compress(raw)
        return base64.b64encode(compressed).decode("utf-8")

    def _invoke_model(self, model: str, compressed_payload: str) -> Dict:
        prompt = self._build_prompt(compressed_payload)
        url = f"{self.base_url}{CHAT_ENDPOINT}"
        body = {
            "model": model,
            "format": "json",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an ensemble analyst. Unpack the compressed payload, "
                        "analyse the features, and return exactly five symbols with "
                        "confidence values between 0 and 1 and concise explanations."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }

        response = requests.post(url, json=body, timeout=self.request_timeout, stream=True)
        if response.status_code != 200:
            raise EnsembleError(f"Ollama {model} rejected request: {response.text}")

        content = self._consume_stream(response)
        data = json.loads(content)
        errors = sorted(VALIDATOR.iter_errors(data), key=lambda e: e.path)
        if errors:
            messages = "; ".join(err.message for err in errors)
            raise EnsembleError(f"Model {model} produced invalid JSON: {messages}")
        return data

    def _consume_stream(self, response: requests.Response) -> str:
        buffer = ""
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line.decode("utf-8"))
            message = chunk.get("message", {})
            buffer += message.get("content", "")
            if chunk.get("done"):
                break
        return buffer

    def _build_prompt(self, compressed_payload: str) -> str:
        return (
            "Payload (base64+zlib compressed JSON):\n"
            f"{compressed_payload}\n\n"
            "Respond strictly with JSON matching:\n"
            '{"choices":[{"symbol":"TICKER","confidence":0.65,"explanation":"..."}]}\n'
            "Return exactly five entries."
        )

    def _check_ollama_available(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def _score_based_fallback(
        self, score_payload: Dict[str, dict]
    ) -> List[EnsembleSelection]:
        """
        Fallback selection based purely on score payload when models are unavailable.

        Selects the top 5 symbols by score. Returns empty list if no scores available.
        """
        if not score_payload:
            logger.warning(
                "No score payload available for fallback selection. "
                "Ensure upstream scoring produces results."
            )
            return []

        scored_items = []
        for symbol, data in score_payload.items():
            score = data.get("score", 0)
            details = data.get("details", {})
            scored_items.append((symbol, score, details))

        scored_items.sort(key=lambda x: x[1], reverse=True)
        top_items = scored_items[:5]

        if not top_items:
            logger.warning("Score payload contained no valid symbols.")
            return []

        if len(top_items) < 5:
            logger.warning(
                "Only %d symbols available (fewer than 5). Proceeding with partial selection.",
                len(top_items),
            )

        selections = []
        for symbol, score, details in top_items:
            # Convert score to confidence (clamp between 0.5 and 0.85)
            confidence = min(0.85, max(0.5, score))
            explanation = (
                f"Score-based selection (no LLM): "
                f"technical={details.get('technical', 0):.2f}, "
                f"sentiment={details.get('sentiment', 0):.2f}, "
                f"momentum={details.get('momentum', 0):.2f}"
            )
            selections.append(
                EnsembleSelection(
                    symbol=symbol,
                    confidence=confidence,
                    explanation=explanation,
                    votes=1,  # Simulated single vote from score engine
                )
            )

        logger.info("Score-based fallback selected: %s", [s.symbol for s in selections])
        return selections


__all__ = ["EnsembleCoordinator", "EnsembleSelection", "EnsembleError"]
