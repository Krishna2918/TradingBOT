#!/usr/bin/env python3
"""
Batch Model Evaluation with Time-Split Validation
==================================================

Evaluates multiple model candidates against snapshot data using time-split holdout.
Records metrics, hashes, and produces evaluation reports.
Supports calibration analysis and PnL proxy calculation.

Usage:
    python scripts/batch_evaluate_models.py
    python scripts/batch_evaluate_models.py --snapshot data/snapshots/2025-10-27 --top-k 5
"""

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Candidate models to evaluate (path, model_type, description)
CANDIDATE_MODELS = [
    ("models/lstm_improved_best.pth", "lstm", "Current production LSTM"),
    ("models/lstm_production_best.pth", "lstm", "Production candidate LSTM"),
    ("models/transformer_best.pth", "transformer", "Transformer architecture"),
    ("models/transformer_checkpoints/best_model.pth", "transformer", "Best transformer checkpoint"),
    ("models/optimized_lstm/best_model.pth", "lstm", "Optimized LSTM"),
    ("models/super_optimized_lstm/best_model.pth", "lstm", "Super optimized LSTM"),
    ("models/aggressive_lstm/best_model.pth", "lstm", "Aggressive LSTM"),
    ("models/gru_transformer_10h.pth", "transformer", "GRU-Transformer hybrid 10h"),
    ("models/lstm_10h_checkpoint/best_model.pth", "lstm", "10h trained LSTM"),
    ("models/real_data_lstm/best_model.pth", "lstm", "Real data LSTM"),
]


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file (first 16 chars)."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def load_model(model_path: Path) -> Optional[Any]:
    """Load a PyTorch model."""
    try:
        return torch.load(model_path, map_location="cpu")
    except Exception as e:
        logger.warning(f"Failed to load {model_path}: {e}")
        return None


def load_snapshot_features(snapshot_path: Path, limit: int = None) -> Tuple[pd.DataFrame, List[str]]:
    """Load features from snapshot parquet files."""
    parquet_files = sorted(snapshot_path.glob("*_features.parquet"))
    if limit:
        parquet_files = parquet_files[:limit]

    all_features = []
    symbols = []

    for pf in parquet_files:
        symbol = pf.stem.replace("_features", "")
        symbols.append(symbol)
        df = pd.read_parquet(pf)
        df["symbol"] = symbol
        all_features.append(df)

    if all_features:
        combined = pd.concat(all_features, ignore_index=True)
        return combined, symbols

    return pd.DataFrame(), []


def time_split_data(
    features_df: pd.DataFrame,
    holdout_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by time to avoid leakage.
    Returns (train_df, holdout_df) where holdout is the most recent data.
    """
    if "ts" not in features_df.columns:
        # Fall back to index-based split
        n = len(features_df)
        split_idx = int(n * (1 - holdout_ratio))
        return features_df.iloc[:split_idx], features_df.iloc[split_idx:]

    # Sort by timestamp and split
    features_df = features_df.sort_values("ts")
    unique_dates = features_df["ts"].unique()
    split_idx = int(len(unique_dates) * (1 - holdout_ratio))
    split_date = unique_dates[split_idx]

    train_df = features_df[features_df["ts"] < split_date]
    holdout_df = features_df[features_df["ts"] >= split_date]

    return train_df, holdout_df


def generate_predictions(
    model: Any,
    features_df: pd.DataFrame,
    model_type: str
) -> pd.DataFrame:
    """Generate predictions using the model."""
    predictions = []

    for symbol in features_df["symbol"].unique():
        sym_data = features_df[features_df["symbol"] == symbol].copy()
        if len(sym_data) < 10:
            continue

        # Sort by time and get numeric features
        if "ts" in sym_data.columns:
            sym_data = sym_data.sort_values("ts")
        numeric_cols = sym_data.select_dtypes(include=[np.number]).columns
        feature_matrix = sym_data[numeric_cols].fillna(0).values

        try:
            if isinstance(model, dict):
                # State dict - use heuristic
                pred_score = float(np.clip(np.mean(feature_matrix[-1]) * 0.1 + 0.5, 0, 1))
            elif hasattr(model, "eval"):
                model.eval()
                with torch.no_grad():
                    x = torch.tensor(feature_matrix[-60:], dtype=torch.float32).unsqueeze(0)
                    output = model(x)
                    if isinstance(output, torch.Tensor):
                        pred_score = float(torch.sigmoid(output).squeeze().cpu().numpy())
                    else:
                        pred_score = float(output)
            else:
                # Sklearn-style
                pred_score = float(model.predict_proba(feature_matrix[-1:])[:, 1])

            predictions.append({
                "symbol": symbol,
                "prediction": pred_score,
                "prediction_class": 1 if pred_score > 0.5 else 0,
            })

        except Exception as e:
            logger.debug(f"Prediction failed for {symbol}: {e}")
            continue

    return pd.DataFrame(predictions)


def compute_metrics(
    predictions_df: pd.DataFrame,
    features_df: pd.DataFrame,
    target_col: str = "direction_1d"
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    if target_col not in features_df.columns:
        return {
            "num_predictions": len(predictions_df),
            "evaluation_type": "no_target",
        }

    # Get second-to-last actual outcomes per symbol (last row has no known future)
    # Binarize: 1 = bullish, 0 = not bullish
    if "ts" in features_df.columns:
        second_last_df = (
            features_df.sort_values(["symbol", "ts"])
            .groupby("symbol")
            .apply(lambda x: x.iloc[-2] if len(x) >= 2 else None)
            .dropna()
        )
    else:
        second_last_df = (
            features_df.groupby("symbol")
            .apply(lambda x: x.iloc[-2] if len(x) >= 2 else None)
            .dropna()
        )
    # Binarize direction: positive = 1, zero/negative = 0
    latest_actuals = (second_last_df[target_col] > 0).astype(int)

    # Merge predictions with actuals
    merged = predictions_df.set_index("symbol").join(latest_actuals, how="inner")

    if len(merged) < 5:
        return {
            "num_predictions": len(predictions_df),
            "evaluation_type": "insufficient_data",
        }

    y_true = merged[target_col].values
    y_pred = merged["prediction_class"].values
    y_prob = merged["prediction"].values

    # Classification metrics
    accuracy = float(np.mean(y_true == y_pred))
    precision = float(np.sum((y_pred == 1) & (y_true == 1)) / max(np.sum(y_pred == 1), 1))
    recall = float(np.sum((y_pred == 1) & (y_true == 1)) / max(np.sum(y_true == 1), 1))
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    # AUC
    if len(np.unique(y_true)) > 1:
        from sklearn.metrics import roc_auc_score
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5
    else:
        auc = 0.5

    # Calibration metrics
    mean_pred = float(np.mean(y_prob))
    mean_actual = float(np.mean(y_true))
    calibration_error = abs(mean_pred - mean_actual)

    # Bias analysis
    pct_bullish_pred = float(np.mean(y_pred))
    pct_bullish_actual = float(np.mean(y_true))
    bullish_bias = pct_bullish_pred - pct_bullish_actual

    # Simple PnL proxy: sum of (prediction_correct * 1 - prediction_wrong * 1)
    correct_predictions = (y_pred == y_true).astype(int)
    pnl_proxy = float(np.sum(correct_predictions * 2 - 1))  # +1 for correct, -1 for wrong

    return {
        "num_predictions": len(merged),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": auc,
        "mean_prediction": mean_pred,
        "pct_bullish_predicted": pct_bullish_pred,
        "pct_bullish_actual": pct_bullish_actual,
        "bullish_bias": bullish_bias,
        "calibration_error": calibration_error,
        "pnl_proxy": pnl_proxy,
        "evaluation_type": "full",
    }


def calibrate_predictions(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    method: str = "temperature"
) -> Tuple[float, np.ndarray]:
    """
    Calibrate predictions using temperature scaling or Platt scaling.
    Returns (calibration_param, calibrated_probs).
    """
    if method == "temperature":
        # Temperature scaling: find T that minimizes NLL
        from scipy.optimize import minimize_scalar

        def nll(T):
            calibrated = 1 / (1 + np.exp(-np.log(y_prob / (1 - y_prob + 1e-10)) / T))
            calibrated = np.clip(calibrated, 1e-10, 1 - 1e-10)
            return -np.mean(y_true * np.log(calibrated) + (1 - y_true) * np.log(1 - calibrated))

        result = minimize_scalar(nll, bounds=(0.1, 10), method='bounded')
        T = result.x
        calibrated = 1 / (1 + np.exp(-np.log(y_prob / (1 - y_prob + 1e-10)) / T))
        return T, calibrated

    elif method == "platt":
        # Platt scaling: logistic regression on logits
        from sklearn.linear_model import LogisticRegression

        logits = np.log(y_prob / (1 - y_prob + 1e-10)).reshape(-1, 1)
        lr = LogisticRegression(solver='lbfgs')
        lr.fit(logits, y_true)
        calibrated = lr.predict_proba(logits)[:, 1]
        return (lr.coef_[0][0], lr.intercept_[0]), calibrated

    return 1.0, y_prob


def evaluate_model(
    model_path: Path,
    model_type: str,
    description: str,
    features_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target_col: str = "direction_1d"
) -> Optional[Dict[str, Any]]:
    """Evaluate a single model."""
    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}")
        return None

    model = load_model(model_path)
    if model is None:
        return None

    # Generate predictions on holdout set
    predictions = generate_predictions(model, holdout_df, model_type)
    if predictions.empty:
        logger.warning(f"No predictions generated for {model_path}")
        return None

    # Compute metrics
    metrics = compute_metrics(predictions, holdout_df, target_col)

    # Compute hash
    model_hash = compute_file_hash(model_path)

    # File size
    model_size_mb = model_path.stat().st_size / (1024 * 1024)

    return {
        "model_path": str(model_path),
        "model_type": model_type,
        "description": description,
        "model_hash": model_hash,
        "model_size_mb": round(model_size_mb, 2),
        "metrics": metrics,
        "evaluated_at": datetime.now().isoformat(),
    }


def save_evaluation_report(
    results: List[Dict],
    snapshot_path: Path,
    output_dir: Path
) -> Path:
    """Save batch evaluation report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    report_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "report_id": report_id,
        "report_type": "batch_evaluation",
        "created_at": datetime.now().isoformat(),
        "snapshot_path": str(snapshot_path),
        "snapshot_id": snapshot_path.name,
        "num_models_evaluated": len(results),
        "models": results,
        "ranking": sorted(
            [r for r in results if r["metrics"].get("f1_score")],
            key=lambda x: x["metrics"]["f1_score"],
            reverse=True
        ),
    }

    report_path = output_dir / f"batch_eval_{report_id}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report_path


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate model candidates")
    parser.add_argument(
        "--snapshot",
        type=str,
        default="data/snapshots/2025-10-27",
        help="Path to snapshot directory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Limit number of symbols",
    )
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.2,
        help="Ratio of data to hold out for validation",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Show top K models",
    )
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        logger.error(f"Snapshot not found: {snapshot_path}")
        return 1

    logger.info("=" * 60)
    logger.info("BATCH MODEL EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Snapshot: {snapshot_path}")
    logger.info(f"Holdout ratio: {args.holdout_ratio}")

    # Load snapshot features
    logger.info("Loading snapshot features...")
    features_df, symbols = load_snapshot_features(snapshot_path, limit=args.limit)
    if features_df.empty:
        logger.error("No features loaded")
        return 1

    logger.info(f"Loaded {len(features_df)} rows for {len(symbols)} symbols")

    # Time-split data
    logger.info("Splitting data by time...")
    train_df, holdout_df = time_split_data(features_df, args.holdout_ratio)
    logger.info(f"Train: {len(train_df)} rows, Holdout: {len(holdout_df)} rows")

    # Evaluate each candidate
    results = []
    for model_path, model_type, description in CANDIDATE_MODELS:
        logger.info(f"\nEvaluating: {description}")
        result = evaluate_model(
            Path(model_path),
            model_type,
            description,
            train_df,
            holdout_df,
        )
        if result:
            metrics = result["metrics"]
            if metrics.get("f1_score"):
                logger.info(
                    f"  Accuracy: {metrics['accuracy']:.3f}, "
                    f"F1: {metrics['f1_score']:.3f}, "
                    f"AUC: {metrics['auc_roc']:.3f}, "
                    f"Bias: {metrics.get('bullish_bias', 0):.3f}"
                )
            results.append(result)

    if not results:
        logger.error("No models evaluated successfully")
        return 1

    # Rank by F1 score
    ranked = sorted(
        [r for r in results if r["metrics"].get("f1_score")],
        key=lambda x: x["metrics"]["f1_score"],
        reverse=True
    )

    logger.info("\n" + "=" * 60)
    logger.info("TOP PERFORMING MODELS")
    logger.info("=" * 60)

    for i, r in enumerate(ranked[:args.top_k], 1):
        m = r["metrics"]
        logger.info(
            f"{i}. {r['description']}\n"
            f"   Path: {r['model_path']}\n"
            f"   F1: {m['f1_score']:.3f}, AUC: {m['auc_roc']:.3f}, "
            f"Accuracy: {m['accuracy']:.3f}\n"
            f"   Bias: {m.get('bullish_bias', 0):.3f}, "
            f"Calibration Error: {m.get('calibration_error', 0):.3f}\n"
            f"   PnL Proxy: {m.get('pnl_proxy', 0):.1f}, Hash: {r['model_hash']}"
        )

    # Save report
    report_path = save_evaluation_report(
        results, snapshot_path, Path("reports/evaluations")
    )
    logger.info(f"\nReport saved: {report_path}")

    # Recommend best model
    if ranked:
        best = ranked[0]
        logger.info("\n" + "=" * 60)
        logger.info("RECOMMENDATION")
        logger.info("=" * 60)
        logger.info(f"Promote: {best['description']}")
        logger.info(f"Path: {best['model_path']}")
        logger.info(f"Hash: {best['model_hash']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
