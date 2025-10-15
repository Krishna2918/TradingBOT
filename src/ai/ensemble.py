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

MODELS = [
    "qwen3-coder:480b-cloud",
    "deepseek-v3.1:671b-cloud",
    "gpt-oss:120b",
]

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
    """

    def __init__(
        self,
        base_url: str = OLLAMA_URL,
        models: Sequence[str] = MODELS,
        request_timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.models = list(models)
        self.request_timeout = request_timeout

    def run(
        self,
        feature_payload: Dict[str, dict],
        score_payload: Dict[str, dict],
    ) -> List[EnsembleSelection]:
        """
        Execute the ensemble and return five selected symbols.
        """

        compressed_payload = self._compress_payload(
            {"features": feature_payload, "scores": score_payload}
        )

        model_outputs = []
        for model in self.models:
            try:
                model_outputs.append(self._invoke_model(model, compressed_payload))
            except Exception as exc:
                logger.exception("Model %s failed: %s", model, exc)
                continue

        if not model_outputs:
            raise EnsembleError("All ensemble models failed to respond.")

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


__all__ = ["EnsembleCoordinator", "EnsembleSelection", "EnsembleError"]
