#!/usr/bin/env python3
"""Utility script to validate connectivity to the configured external APIs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import requests


ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env_values(env_path: Path = ENV_FILE) -> Dict[str, str]:
    """Load key/value pairs from a .env file into a dictionary.

    The repository does not depend on python-dotenv at runtime, so we
    implement a tiny parser here to avoid adding new dependencies.
    """

    values: Dict[str, str] = {}
    if not env_path.exists():
        return values

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def request_with_handling(url: str, **kwargs) -> Tuple[bool, str, Dict[str, object]]:
    """Perform a GET request and capture a success flag and message."""
    try:
        response = requests.get(url, timeout=15, **kwargs)
        body: Dict[str, object]
        try:
            body = response.json()
        except ValueError:
            body = {"raw": response.text[:200]}
        if response.ok:
            return True, f"HTTP {response.status_code}", body
        return False, f"HTTP {response.status_code}", body
    except requests.RequestException as exc:  # pragma: no cover - network failure
        return False, f"Request error: {exc}", {}


def validate_alpha_vantage(api_key: str) -> Tuple[bool, Dict[str, object]]:
    url = (
        "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey="
        f"{api_key}"
    )
    success, message, body = request_with_handling(url)
    is_valid = success and "Global Quote" in body and body["Global Quote"]
    return bool(is_valid), {"message": message, "payload": body}


def validate_news_api(api_key: str) -> Tuple[bool, Dict[str, object]]:
    url = (
        "https://newsapi.org/v2/everything?q=apple&sortBy=publishedAt&apiKey="
        f"{api_key}"
    )
    success, message, body = request_with_handling(url)
    articles = body.get("articles") if isinstance(body, dict) else None
    is_valid = success and isinstance(articles, list)
    return bool(is_valid), {
        "message": message,
        "article_count": len(articles) if isinstance(articles, list) else 0,
        "payload": body,
    }


def validate_finnhub(api_key: str) -> Tuple[bool, Dict[str, object]]:
    url = f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={api_key}"
    success, message, body = request_with_handling(url)
    is_valid = success and isinstance(body, dict) and "c" in body
    return bool(is_valid), {"message": message, "payload": body}


def validate_questrade(refresh_token: str) -> Tuple[bool, Dict[str, object]]:
    url = (
        "https://login.questrade.com/oauth2/token?grant_type=refresh_token&refresh_token="
        f"{refresh_token}"
    )
    success, message, body = request_with_handling(url)
    # Successful responses include an access_token field. We intentionally do not
    # surface the token value to avoid leaking secrets in logs.
    is_valid = success and isinstance(body, dict) and "access_token" in body
    sanitized_payload: Dict[str, object] = {}
    if isinstance(body, dict):
        sanitized_payload = {key: ("<redacted>" if key.endswith("token") else value) for key, value in body.items()}
    return bool(is_valid), {"message": message, "payload": sanitized_payload}


def main() -> None:
    env_values = load_env_values()

    required_keys = {
        "ALPHA_VANTAGE_API_KEY": validate_alpha_vantage,
        "NEWSAPI_KEY": validate_news_api,
        "FINNHUB_API_KEY": validate_finnhub,
        "QUESTRADE_REFRESH_TOKEN": validate_questrade,
    }

    missing = [key for key in required_keys if key not in env_values]
    if missing:
        print("⚠️ Missing environment values:", ", ".join(missing))
        print("Please update .env before running connectivity tests.")
        return

    results = {}
    for key, validator in required_keys.items():
        print(f"\n🔍 Testing {key}...")
        success, details = validator(env_values[key])
        results[key] = {"success": success, **details}
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {details['message']}")
        if details.get("payload"):
            print(json.dumps(details["payload"], indent=2)[:1000])

    print("\n📊 Summary:")
    for key, info in results.items():
        status = "✅" if info["success"] else "❌"
        print(f"{status} {key}")


if __name__ == "__main__":
    main()
