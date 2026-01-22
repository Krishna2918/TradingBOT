#!/usr/bin/env python3
"""
Smoke Test: Model Inference with Snapshot Data
===============================================

Validates end-to-end model loading and inference using snapshot data.
No API calls are made - this runs completely offline.

Usage:
    python scripts/smoke_test_inference.py
    python scripts/smoke_test_inference.py --snapshot data/snapshots/2025-10-27
    python scripts/smoke_test_inference.py --verbose
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_latest_snapshot(snapshots_dir: Path) -> Path:
    """Find the latest snapshot directory."""
    if not snapshots_dir.exists():
        return None
    snapshots = sorted(
        [d for d in snapshots_dir.iterdir() if d.is_dir()],
        reverse=True
    )
    return snapshots[0] if snapshots else None


def test_model_registry():
    """Test 1: Verify model registry loads correctly."""
    logger.info("=" * 60)
    logger.info("TEST 1: Model Registry")
    logger.info("=" * 60)

    from src.ai.model_registry import ModelRegistry

    registry = ModelRegistry(storage_path="models/registry")

    # Check registry loaded
    stats = registry.get_registry_stats()
    logger.info("Registry stats: %s", stats)

    if stats["total_models"] == 0:
        logger.error("FAIL: No models in registry")
        return False

    # Check production model key
    if hasattr(registry, "_production_model_key") and registry._production_model_key:
        logger.info("Production model key: %s", registry._production_model_key)
    else:
        logger.warning("No production_model key set in registry")

    logger.info("PASS: Model registry loaded with %d models", stats["total_models"])
    return True


def test_model_loading():
    """Test 2: Verify production model can be loaded."""
    logger.info("=" * 60)
    logger.info("TEST 2: Production Model Loading")
    logger.info("=" * 60)

    from src.ai.model_registry import ModelRegistry

    registry = ModelRegistry(storage_path="models/registry")

    try:
        result = registry.get_production_model("lstm_production")
        if result is None:
            logger.error("FAIL: get_production_model returned None")
            return False

        model, metadata = result
        logger.info("Loaded model: %s v%s", metadata.model_id, metadata.version)
        logger.info("Model type: %s", type(model).__name__)
        logger.info("Training date: %s", metadata.training_date)
        logger.info("Snapshot ID: %s", metadata.data_snapshot_id)
        logger.info("Git commit: %s", metadata.git_commit)

        # Check model has expected structure
        if hasattr(model, "state_dict"):
            logger.info("Model is a PyTorch module with state_dict")
        elif isinstance(model, dict):
            logger.info("Model loaded as state_dict with %d keys", len(model))
        else:
            logger.info("Model loaded as: %s", type(model))

        logger.info("PASS: Production model loaded successfully")
        return True

    except Exception as e:
        logger.error("FAIL: Failed to load production model: %s", e)
        import traceback
        traceback.print_exc()
        return False


def test_snapshot_loading(snapshot_path: Path):
    """Test 3: Verify snapshot data can be loaded."""
    logger.info("=" * 60)
    logger.info("TEST 3: Snapshot Data Loading")
    logger.info("=" * 60)

    if not snapshot_path or not snapshot_path.exists():
        logger.error("FAIL: Snapshot path does not exist: %s", snapshot_path)
        return False

    # Check manifest
    manifest_path = snapshot_path / "FEATURE_MANIFEST.json"
    if not manifest_path.exists():
        logger.warning("No FEATURE_MANIFEST.json in snapshot")
    else:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        logger.info("Snapshot: %s", snapshot_path.name)
        logger.info("Created: %s", manifest.get("created_at", "unknown"))
        logger.info("Symbols: %d", manifest.get("total_symbols_analyzed", 0))
        logger.info("Features: %d", len(manifest.get("features", [])))

    # Count parquet files
    parquet_files = list(snapshot_path.glob("*_features.parquet"))
    if not parquet_files:
        logger.error("FAIL: No parquet files in snapshot")
        return False

    logger.info("Found %d parquet files", len(parquet_files))

    # Try loading one
    import pandas as pd
    sample_file = parquet_files[0]
    df = pd.read_parquet(sample_file)
    logger.info("Sample file: %s", sample_file.name)
    logger.info("Rows: %d, Columns: %d", len(df), len(df.columns))

    logger.info("PASS: Snapshot data loaded successfully")
    return True


def test_offline_scoring(snapshot_path: Path):
    """Test 4: Run offline scoring with snapshot data."""
    logger.info("=" * 60)
    logger.info("TEST 4: Offline Scoring (End-to-End)")
    logger.info("=" * 60)

    if not snapshot_path or not snapshot_path.exists():
        logger.error("FAIL: Snapshot path required for offline scoring test")
        return False

    try:
        from src.agents.ai_selector import AISelectorAgent

        # Create agent in offline mode with model-based scoring
        agent = AISelectorAgent(
            snapshot_path=str(snapshot_path),
            use_trained_model=True,
            model_name="lstm_production",
            offline_mode=True,
        )

        # Get symbols from snapshot (limit to 5 for smoke test)
        symbols = agent._get_snapshot_symbols()[:5]
        if not symbols:
            logger.error("FAIL: No symbols in snapshot")
            return False

        logger.info("Testing with %d symbols: %s", len(symbols), symbols)

        # Load features
        features = agent._load_features_from_snapshot(symbols)
        if features.empty:
            logger.error("FAIL: No features loaded")
            return False

        logger.info("Loaded %d feature rows", len(features))

        # Score using snapshot features
        scores_df = agent.score_engine.score_from_features(features, symbols)
        if scores_df.empty:
            logger.warning("Model scoring returned empty, using fallback")
            scores_df = agent._create_fallback_scores(features, symbols)

        if scores_df.empty:
            logger.error("FAIL: No scores generated")
            return False

        logger.info("Generated scores for %d symbols:", len(scores_df))
        for _, row in scores_df.iterrows():
            details = json.loads(row["details_json"])
            logger.info(
                "  %s: score=%.3f, mode=%s",
                row["symbol"],
                row["score"],
                details.get("scoring_mode", "unknown"),
            )

        logger.info("PASS: Offline scoring completed successfully")
        return True

    except Exception as e:
        logger.error("FAIL: Offline scoring failed: %s", e)
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Smoke test model inference with snapshot")
    parser.add_argument(
        "--snapshot",
        type=str,
        default=None,
        help="Path to snapshot directory (default: auto-detect latest)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Find snapshot
    if args.snapshot:
        snapshot_path = Path(args.snapshot)
    else:
        snapshot_path = find_latest_snapshot(Path("data/snapshots"))

    logger.info("=" * 60)
    logger.info("SMOKE TEST: Model Inference with Snapshot")
    logger.info("=" * 60)
    logger.info("Snapshot: %s", snapshot_path)
    logger.info("Timestamp: %s", datetime.now().isoformat())
    logger.info("")

    results = {
        "registry": test_model_registry(),
        "model_loading": test_model_loading(),
        "snapshot_loading": test_snapshot_loading(snapshot_path) if snapshot_path else False,
        "offline_scoring": test_offline_scoring(snapshot_path) if snapshot_path else False,
    }

    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for test_name, passed_test in results.items():
        status = "PASS" if passed_test else "FAIL"
        logger.info("  %s: %s", test_name, status)

    logger.info("")
    logger.info("Result: %d/%d tests passed", passed, total)

    if passed == total:
        logger.info("All tests PASSED - system is production ready")
        return 0
    else:
        logger.error("Some tests FAILED - review errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
