"""Entry point for the AI Trading System scaffold."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Callable, Optional

from src.config import get_settings
from src.workflow import DailyWorkflow

# --- Always-on AI switch ---
AI_MODE_DEFAULT = True  # set to False to disable globally


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the trading controller."""

    parser = argparse.ArgumentParser(description="AI Trading System controller")
    parser.add_argument("--demo", action="store_true", help="Run in demo (paper trading) mode")
    parser.add_argument("--dry-run", action="store_true", help="Simulate workflow without executing trades")
    parser.add_argument("--today", action="store_true", help="Limit processing to the current trading day")
    parser.add_argument("--ai", action="store_true", help="Enable AI ensemble scoring")
    return parser.parse_args()


def resolve_ai_scoring() -> Optional[Callable[[], None]]:
    """Locate the ensemble scoring entry point if it exists."""

    try:
        from src.agents.ai_selector import run_ai_scoring as scorer  # type: ignore

        return scorer
    except ModuleNotFoundError:
        pass
    except Exception as exc:
        print(f"AI scoring import error (agents): {exc}")
        return None

    try:
        from src.ai.selector import run_ai_scoring as scorer  # type: ignore

        return scorer
    except ModuleNotFoundError:
        return None
    except Exception as exc:
        print(f"AI scoring import error (ai): {exc}")
        return None


def main() -> None:
    args = parse_args()
    ai_mode = AI_MODE_DEFAULT or args.ai

    settings = get_settings()

    if ai_mode:
        print("AI Mode: ENABLED - running ensemble scoring & daily selection (5 picks)...")
        scorer = resolve_ai_scoring()
        if scorer is not None:
            try:
                scorer()
            except Exception as exc:
                print(f"AI scoring execution failed: {exc}")
        else:
            print("AI scoring module not found; skipping ensemble scoring step.")

    workflow = DailyWorkflow(settings=settings)
    result = workflow.run(demo=args.demo, dry_run=args.dry_run, today=args.today)

    print("AI Trading System scaffold executed")
    print("Runtime configuration:")
    settings_payload = settings.model_dump()

    for key, value in settings_payload.items():
        print(f"  {key}: {value}")

    print("\nWorkflow summary:")
    for key, value in asdict(result).items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
