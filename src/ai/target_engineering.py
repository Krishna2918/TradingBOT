"""
Target Engineering for Trading Models

Consistent target creation with configurable neutral bands to address class imbalance.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def build_direction_targets(close: pd.Series, 
                          horizon: int = 1, 
                          neutral_band: float = 0.005) -> pd.Series:
    """
    Build consistent direction targets from price data.
    
    Args:
        close: Close price series
        horizon: Number of days to look forward (default: 1)
        neutral_band: Neutral band threshold (default: 0.005 = ±0.5%)
        
    Returns:
        Series with direction labels: -1=Down, 0=Flat, 1=Up
    """
    # Forward return over `horizon` days
    fwd_ret = (close.shift(-horizon) / close) - 1.0
    
    # Label: -1=Down, 0=Flat, 1=Up
    y = np.where(fwd_ret > neutral_band, 1,
                np.where(fwd_ret < -neutral_band, -1, 0))
    
    return pd.Series(y, index=close.index, name="direction_1d")

def analyze_target_distribution(targets: pd.Series, symbol: str = "Unknown") -> dict:
    """
    Analyze target distribution and return statistics.
    
    Args:
        targets: Target series with values -1, 0, 1
        symbol: Symbol name for logging
        
    Returns:
        Dictionary with distribution statistics
    """
    value_counts = targets.value_counts().sort_index()
    total = len(targets)
    
    stats = {
        'symbol': symbol,
        'total_samples': total,
        'down_count': value_counts.get(-1, 0),
        'flat_count': value_counts.get(0, 0),
        'up_count': value_counts.get(1, 0),
        'down_pct': (value_counts.get(-1, 0) / total) * 100,
        'flat_pct': (value_counts.get(0, 0) / total) * 100,
        'up_pct': (value_counts.get(1, 0) / total) * 100
    }
    
    logger.info(f"{symbol} target distribution:")
    logger.info(f"  DOWN (-1): {stats['down_count']:,} ({stats['down_pct']:.1f}%)")
    logger.info(f"  FLAT (0):  {stats['flat_count']:,} ({stats['flat_pct']:.1f}%)")
    logger.info(f"  UP (1):    {stats['up_count']:,} ({stats['up_pct']:.1f}%)")
    
    return stats

def rebuild_targets_for_symbol(df: pd.DataFrame, 
                             horizon: int = 1, 
                             neutral_band: float = 0.005,
                             symbol: str = "Unknown") -> pd.DataFrame:
    """
    Rebuild direction_1d targets for a single symbol DataFrame.
    
    Args:
        df: DataFrame with 'close' column
        horizon: Forward horizon in days
        neutral_band: Neutral band threshold
        symbol: Symbol name for logging
        
    Returns:
        DataFrame with updated direction_1d column
    """
    if 'close' not in df.columns:
        raise ValueError(f"DataFrame for {symbol} missing 'close' column")
    
    # Build new targets
    new_targets = build_direction_targets(df['close'], horizon, neutral_band)
    
    # Update DataFrame
    df = df.copy()
    df['direction_1d'] = new_targets
    
    # Analyze distribution
    analyze_target_distribution(new_targets, symbol)
    
    return df

def encode_targets_for_training(targets: pd.Series) -> np.ndarray:
    """
    Encode targets for neural network training.
    
    Args:
        targets: Series with values -1, 0, 1
        
    Returns:
        Numpy array with values 0, 1, 2 for CrossEntropy loss
    """
    # Convert -1,0,1 -> 0,1,2
    encoded = targets.to_numpy().astype(np.int8) + 1
    return encoded

def create_balanced_sampler_weights(y_encoded: np.ndarray) -> np.ndarray:
    """
    Create sample weights for balanced training.
    
    Args:
        y_encoded: Encoded targets (0, 1, 2)
        
    Returns:
        Sample weights for WeightedRandomSampler
    """
    # Calculate class counts
    class_counts = np.bincount(y_encoded, minlength=3)
    
    # Calculate class weights (inverse frequency)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    
    # Map to sample weights
    sample_weights = class_weights[y_encoded]
    
    logger.info(f"Class counts: DOWN={class_counts[0]}, FLAT={class_counts[1]}, UP={class_counts[2]}")
    logger.info(f"Class weights: DOWN={class_weights[0]:.4f}, FLAT={class_weights[1]:.4f}, UP={class_weights[2]:.4f}")
    
    return sample_weights

def validate_no_leakage(X_seq_end_idx: int, y_target_idx: int, symbol: str = "Unknown"):
    """
    Validate that there's no data leakage in sequence creation.
    
    Args:
        X_seq_end_idx: Last index used in feature sequence
        y_target_idx: Index of target label
        symbol: Symbol name for error reporting
    """
    if X_seq_end_idx >= y_target_idx:
        raise ValueError(f"Data leakage detected in {symbol}: "
                        f"Feature sequence ends at {X_seq_end_idx} "
                        f"but target is at {y_target_idx}")

def compute_macro_f1_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute macro-F1 score for 3-class classification.
    
    Args:
        y_true: True labels (0, 1, 2)
        y_pred: Predicted labels (0, 1, 2)
        
    Returns:
        Macro-F1 score
    """
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

def log_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: list = None) -> dict:
    """
    Log detailed per-class metrics.
    
    Args:
        y_true: True labels (0, 1, 2)
        y_pred: Predicted labels (0, 1, 2)
        class_names: Optional class names for logging
        
    Returns:
        Dictionary with per-class metrics
    """
    from sklearn.metrics import classification_report, confusion_matrix
    
    if class_names is None:
        class_names = ['DOWN (0)', 'FLAT (1)', 'UP (2)']
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                 zero_division=0, output_dict=True)
    
    # Log per-class metrics
    for i, class_name in enumerate(class_names):
        if str(i) in report:
            metrics = report[str(i)]
            logger.info(f"{class_name}: F1={metrics['f1-score']:.3f}, "
                       f"Precision={metrics['precision']:.3f}, "
                       f"Recall={metrics['recall']:.3f}")
    
    # Log aggregate metrics
    macro_f1 = report['macro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']
    logger.info(f"Macro F1: {macro_f1:.4f}")
    logger.info(f"Weighted F1: {weighted_f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    return {
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_metrics': {str(i): report[str(i)] for i in range(3) if str(i) in report},
        'confusion_matrix': cm.tolist()
    }