#!/usr/bin/env python3
"""
Command-line entry point for the AI trading workflow.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent))

from src.agents.ai_selector import AISelectorAgent  # noqa: E402
from src.dashboard.connector import DashboardConnector  # noqa: E402


LOG_PATH = Path("logs/system.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=handlers,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AI trading pipeline.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode (paper execution only).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1200,
        help="Maximum number of symbols to evaluate prior to filtering.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def run_pipeline(limit: int) -> None:
    logger = logging.getLogger("pipeline")
    agent = AISelectorAgent()
    logger.info("Universe ingestion + feature engineering")

    selections = agent.run(limit=limit)
    if not selections:
        logger.warning("Pipeline completed with no selections.")
        return

    dash = DashboardConnector()
    recent_picks = dash.ai_picks().head(5)
    logger.info("Latest AI picks:\n%s", recent_picks.to_string(index=False))
    logger.info("Risk events (last 5):\n%s", dash.risk_events(limit=5))


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    load_dotenv(dotenv_path=".env")

    logging.getLogger("main").info("AI trading pipeline starting (demo=%s)", args.demo)
    run_pipeline(args.limit)
    logging.getLogger("main").info("Pipeline run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
