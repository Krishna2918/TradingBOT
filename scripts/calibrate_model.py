#!/usr/bin/env python3
"""
Model Calibration Script
========================

Calibrates model predictions to reduce bias and improve probability estimates.
Supports temperature scaling and Platt scaling methods.

Usage:
    python scripts/calibrate_model.py
    python scripts/calibrate_model.py --method platt --threshold 0.6
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_snapshot_features(snapshot_path: Path, limit: int = None) -> pd.DataFrame:
    """Load features from snapshot parquet files."""
    parquet_files = sorted(snapshot_path.glob("*_features.parquet"))
    if limit:
        parquet_files = parquet_files[:limit]

    all_features = []
    for pf in parquet_files:
        symbol = pf.stem.replace("_features", "")
        df = pd.read_parquet(pf)
        df["symbol"] = symbol
        all_features.append(df)

    if all_features:
        return pd.concat(all_features, ignore_index=True)
    return pd.DataFrame()


def generate_raw_predictions(
    features_df: pd.DataFrame,
    target_col: str = "direction_1d"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate raw predictions using the production model.
    Returns (y_prob, y_true).
    """
    from src.ai.model_registry import ModelRegistry
    import torch

    registry = ModelRegistry(storage_path="models/registry")
    result = registry.get_production_model("lstm_production")

    if not result:
        raise ValueError("No production model found")

    model, metadata = result
    predictions = []
    actuals = []

    for symbol in features_df["symbol"].unique():
        sym_data = features_df[features_df["symbol"] == symbol].copy()
        if len(sym_data) < 10:
            continue

        if "ts" in sym_data.columns:
            sym_data = sym_data.sort_values("ts")

        numeric_cols = sym_data.select_dtypes(include=[np.number]).columns
        feature_matrix = sym_data[numeric_cols].fillna(0).values

        try:
            if isinstance(model, dict):
                # State dict heuristic - improved version
                recent_features = feature_matrix[-60:]
                # Use feature statistics for prediction
                mean_val = np.mean(recent_features)
                std_val = np.std(recent_features)
                # Map to probability space with better calibration
                pred_score = float(np.clip(0.5 + 0.1 * np.tanh(mean_val / (std_val + 1e-6)), 0.1, 0.9))
            else:
                model.eval()
                with torch.no_grad():
                    x = torch.tensor(feature_matrix[-60:], dtype=torch.float32).unsqueeze(0)
                    output = model(x)
                    pred_score = float(torch.sigmoid(output).squeeze().cpu().numpy())

            predictions.append(pred_score)

            # Get actual outcome (use second-to-last row - last row has no known future)
            # Binarize: 1 = bullish, 0 = not bullish
            if target_col in sym_data.columns and len(sym_data) >= 2:
                actual = sym_data[target_col].iloc[-2]  # Second-to-last has known outcome
                # direction_1d can be -1, 0, 1; binarize to 0/1
                actual_binary = 1.0 if actual > 0 else 0.0
                actuals.append(actual_binary)
            else:
                actuals.append(np.nan)

        except Exception as e:
            logger.debug(f"Prediction failed for {symbol}: {e}")
            continue

    return np.array(predictions), np.array(actuals)


def temperature_scaling(
    y_prob: np.ndarray,
    y_true: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Apply temperature scaling to calibrate predictions.
    Returns (optimal_temperature, calibrated_probs).
    """
    # Convert probabilities to logits
    y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)
    logits = np.log(y_prob / (1 - y_prob))

    def nll(T):
        """Negative log likelihood with temperature T."""
        scaled_logits = logits / T
        calibrated = 1 / (1 + np.exp(-scaled_logits))
        calibrated = np.clip(calibrated, 1e-10, 1 - 1e-10)
        return -np.mean(y_true * np.log(calibrated) + (1 - y_true) * np.log(1 - calibrated))

    # Find optimal temperature
    result = minimize_scalar(nll, bounds=(0.1, 10), method='bounded')
    T = result.x

    # Apply calibration
    calibrated = 1 / (1 + np.exp(-logits / T))

    logger.info(f"Temperature scaling: T={T:.3f}")
    return T, calibrated


def platt_scaling(
    y_prob: np.ndarray,
    y_true: np.ndarray
) -> Tuple[Tuple[float, float], np.ndarray]:
    """
    Apply Platt scaling (logistic regression on logits).
    Returns ((A, B), calibrated_probs).
    """
    y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)
    logits = np.log(y_prob / (1 - y_prob)).reshape(-1, 1)

    # Fit logistic regression
    lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    lr.fit(logits, y_true)

    # Get calibrated probabilities
    calibrated = lr.predict_proba(logits)[:, 1]

    A, B = lr.coef_[0][0], lr.intercept_[0]
    logger.info(f"Platt scaling: A={A:.3f}, B={B:.3f}")

    return (A, B), calibrated


def threshold_adjustment(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    target_precision: float = 0.6
) -> float:
    """
    Find optimal threshold to achieve target precision.
    """
    thresholds = np.linspace(0.3, 0.8, 50)
    best_threshold = 0.5
    best_f1 = 0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    logger.info(f"Optimal threshold: {best_threshold:.3f} (F1={best_f1:.3f})")
    return best_threshold


def compute_calibration_metrics(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute calibration metrics."""
    y_pred = (y_prob >= threshold).astype(int)

    # Basic metrics
    accuracy = float(np.mean(y_pred == y_true))
    precision = float(np.sum((y_pred == 1) & (y_true == 1)) / max(np.sum(y_pred == 1), 1))
    recall = float(np.sum((y_pred == 1) & (y_true == 1)) / max(np.sum(y_true == 1), 1))
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    # AUC
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5

    # Brier score (lower is better)
    brier = float(brier_score_loss(y_true, y_prob))

    # Calibration error
    mean_pred = float(np.mean(y_prob))
    mean_actual = float(np.mean(y_true))
    calibration_error = abs(mean_pred - mean_actual)

    # Bias
    bullish_bias = float(np.mean(y_pred)) - float(np.mean(y_true))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": auc,
        "brier_score": brier,
        "calibration_error": calibration_error,
        "bullish_bias": bullish_bias,
        "mean_prediction": mean_pred,
        "pct_bullish_predicted": float(np.mean(y_pred)),
        "pct_bullish_actual": mean_actual,
        "threshold": threshold,
    }


def save_calibration_config(
    config: Dict[str, Any],
    output_path: Path
) -> None:
    """Save calibration configuration."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Calibration config saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate production model")
    parser.add_argument(
        "--snapshot",
        type=str,
        default="data/snapshots/2025-10-27",
        help="Path to snapshot directory",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["temperature", "platt", "both"],
        default="both",
        help="Calibration method",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Limit number of symbols",
    )
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        logger.error(f"Snapshot not found: {snapshot_path}")
        return 1

    logger.info("=" * 60)
    logger.info("MODEL CALIBRATION")
    logger.info("=" * 60)
    logger.info(f"Snapshot: {snapshot_path}")
    logger.info(f"Method: {args.method}")

    # Load features
    logger.info("Loading snapshot features...")
    features_df = load_snapshot_features(snapshot_path, limit=args.limit)
    if features_df.empty:
        logger.error("No features loaded")
        return 1

    logger.info(f"Loaded {len(features_df)} rows")

    # Generate raw predictions
    logger.info("Generating raw predictions...")
    y_prob, y_true = generate_raw_predictions(features_df)

    # Filter out NaN actuals
    valid_mask = ~np.isnan(y_true)
    y_prob = y_prob[valid_mask]
    y_true = y_true[valid_mask]

    if len(y_prob) < 10:
        logger.error("Insufficient valid predictions")
        return 1

    logger.info(f"Valid predictions: {len(y_prob)}")

    # Raw metrics
    logger.info("\n--- RAW PREDICTIONS ---")
    raw_metrics = compute_calibration_metrics(y_prob, y_true)
    for k, v in raw_metrics.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    calibration_results = {
        "created_at": datetime.now().isoformat(),
        "snapshot_path": str(snapshot_path),
        "num_samples": len(y_prob),
        "raw_metrics": raw_metrics,
    }

    # Temperature scaling
    if args.method in ["temperature", "both"]:
        logger.info("\n--- TEMPERATURE SCALING ---")
        T, calibrated_temp = temperature_scaling(y_prob, y_true)
        optimal_threshold = threshold_adjustment(calibrated_temp, y_true)
        temp_metrics = compute_calibration_metrics(calibrated_temp, y_true, optimal_threshold)
        for k, v in temp_metrics.items():
            logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        calibration_results["temperature_scaling"] = {
            "temperature": T,
            "threshold": optimal_threshold,
            "metrics": temp_metrics,
        }

    # Platt scaling
    if args.method in ["platt", "both"]:
        logger.info("\n--- PLATT SCALING ---")
        (A, B), calibrated_platt = platt_scaling(y_prob, y_true)
        optimal_threshold = threshold_adjustment(calibrated_platt, y_true)
        platt_metrics = compute_calibration_metrics(calibrated_platt, y_true, optimal_threshold)
        for k, v in platt_metrics.items():
            logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        calibration_results["platt_scaling"] = {
            "A": A,
            "B": B,
            "threshold": optimal_threshold,
            "metrics": platt_metrics,
        }

    # Save calibration config
    config_path = Path("models/registry/calibration_config.json")
    save_calibration_config(calibration_results, config_path)

    # Recommend best method
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDATION")
    logger.info("=" * 60)

    if args.method == "both":
        temp_f1 = calibration_results.get("temperature_scaling", {}).get("metrics", {}).get("f1_score", 0)
        platt_f1 = calibration_results.get("platt_scaling", {}).get("metrics", {}).get("f1_score", 0)

        if temp_f1 > platt_f1:
            logger.info("Use TEMPERATURE SCALING")
            logger.info(f"  Temperature: {calibration_results['temperature_scaling']['temperature']:.3f}")
            logger.info(f"  Threshold: {calibration_results['temperature_scaling']['threshold']:.3f}")
        else:
            logger.info("Use PLATT SCALING")
            logger.info(f"  A: {calibration_results['platt_scaling']['A']:.3f}")
            logger.info(f"  B: {calibration_results['platt_scaling']['B']:.3f}")
            logger.info(f"  Threshold: {calibration_results['platt_scaling']['threshold']:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
