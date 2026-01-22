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
        default=50,
        help="Maximum number of symbols to evaluate prior to filtering (default: 50 to avoid API rate limits).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    # Offline/snapshot mode
    parser.add_argument(
        "--use-snapshot",
        type=str,
        default=None,
        metavar="PATH",
        help="Run in offline mode using data snapshot (e.g., data/snapshots/2025-10-27). No API calls.",
    )
    parser.add_argument(
        "--offline-mode",
        action="store_true",
        help="Run in offline mode using static universe (no live API calls).",
    )
    # Model-based scoring
    parser.add_argument(
        "--use-model",
        action="store_true",
        help="Use trained ML model for scoring instead of rule-based scoring.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="lstm_production",
        help="Name of the model to use (default: lstm_production).",
    )
    return parser.parse_args()


def run_pipeline(
    limit: int,
    snapshot_path: str = None,
    offline_mode: bool = False,
    use_model: bool = False,
    model_name: str = "lstm_production",
) -> None:
    logger = logging.getLogger("pipeline")

    # Initialize agent with mode settings
    agent = AISelectorAgent(
        snapshot_path=snapshot_path,
        offline_mode=offline_mode or (snapshot_path is not None),
        use_trained_model=use_model,
        model_name=model_name if use_model else None,
    )

    mode_desc = []
    if snapshot_path:
        mode_desc.append(f"snapshot={snapshot_path}")
    if offline_mode or snapshot_path:
        mode_desc.append("offline")
    if use_model:
        mode_desc.append(f"model={model_name}")
    else:
        mode_desc.append("rule-based")

    logger.info("Starting pipeline: %s", ", ".join(mode_desc) if mode_desc else "live+rule-based")

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

    logger = logging.getLogger("main")
    logger.info(
        "AI trading pipeline starting (demo=%s, offline=%s, use_model=%s)",
        args.demo,
        args.offline_mode or bool(args.use_snapshot),
        args.use_model,
    )

    run_pipeline(
        limit=args.limit,
        snapshot_path=args.use_snapshot,
        offline_mode=args.offline_mode,
        use_model=args.use_model,
        model_name=args.model_name,
    )

    logger.info("Pipeline run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
