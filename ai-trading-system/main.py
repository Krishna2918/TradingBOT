"""Entry point for the AI Trading System scaffold."""

from __future__ import annotations

import argparse
from dataclasses import asdict

from src.config import get_settings
from src.workflow import DailyWorkflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Trading System controller")
    parser.add_argument("--demo", action="store_true", help="Run in demo (paper trading) mode")
    parser.add_argument("--dry-run", action="store_true", help="Simulate workflow without executing trades")
    parser.add_argument("--today", action="store_true", help="Limit processing to the current trading day")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    workflow = DailyWorkflow(settings=settings)
    result = workflow.run(demo=args.demo, dry_run=args.dry_run, today=args.today)

    print("AI Trading System scaffold executed")
    print("Runtime configuration:")
    try:
        settings_payload = settings.dict()
    except AttributeError:  # Pydantic v2
        settings_payload = settings.model_dump()

    for key, value in settings_payload.items():
        print(f"  {key}: {value}")

    print("\nWorkflow summary:")
    for key, value in asdict(result).items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
