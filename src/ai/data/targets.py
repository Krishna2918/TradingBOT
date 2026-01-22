"""
Consistent target creation for trading models.

This module ensures all symbols get the same target creation logic,
preventing synthetic target fallbacks and ensuring reproducible labels.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def ensure_direction_1d(df: pd.DataFrame,
                       close_col: str = "close",
                       neutral_band: float = 0.004,  # ±0.4%
                       symbol: str = "Unknown") -> pd.DataFrame:
    """
    Adds 'direction_1d' if missing, using *forward* 1-day return and a neutral band.
    
    Labels: -1=DOWN, 0=FLAT, 1=UP
    
    Args:
        df: DataFrame with price data
        close_col: Name of close price column
        neutral_band: Neutral band threshold (e.g., 0.004 = ±0.4%)
        symbol: Symbol name for logging
        
    Returns:
        DataFrame with direction_1d column
        
    Raises:
        ValueError: If close column is missing
    """
    if "direction_1d" in df.columns:
        # Already present, validate and return
        _validate_existing_targets(df, symbol)
        return df
    
    if close_col not in df.columns:
        raise ValueError(f"Missing close column '{close_col}' needed to build targets for {symbol}")
    
    logger.info(f"Building direction_1d targets for {symbol} with neutral band ±{neutral_band*100:.1f}%")
    
    # Forward return: compare tomorrow's close to today's close (avoid leakage)
    fwd_ret = df[close_col].shift(-1) / df[close_col] - 1.0
    
    # Label with neutral band
    y = np.where(fwd_ret > neutral_band, 1,
                np.where(fwd_ret < -neutral_band, -1, 0))
    
    df = df.copy()
    df["direction_1d"] = y
    
    # Log distribution
    _log_target_distribution(df, symbol)
    
    # Optional: Remove last row with NaN forward return
    # df = df.iloc[:-1]  # Uncomment if you want to strictly remove lookahead row
    
    return df

def _validate_existing_targets(df: pd.DataFrame, symbol: str) -> None:
    """Validate existing direction_1d targets."""
    targets = df["direction_1d"].dropna()
    unique_vals = set(targets.unique())
    expected_vals = {-1, 0, 1}
    
    if not unique_vals.issubset(expected_vals):
        invalid_vals = unique_vals - expected_vals
        raise ValueError(f"{symbol}: Invalid direction_1d values found: {invalid_vals}")
    
    # Check for missing classes (warning, not error)
    missing_classes = expected_vals - unique_vals
    if missing_classes:
        logger.warning(f"{symbol}: Missing target classes: {missing_classes}")

def _log_target_distribution(df: pd.DataFrame, symbol: str) -> None:
    """Log target distribution for monitoring."""
    targets = df["direction_1d"].dropna()
    if len(targets) == 0:
        logger.warning(f"{symbol}: No valid targets found")
        return
    
    vals, counts = np.unique(targets.values, return_counts=True)
    total = len(targets)
    
    distribution = {}
    for val, count in zip(vals, counts):
        pct = (count / total) * 100
        distribution[int(val)] = f"{count} ({pct:.1f}%)"
    
    logger.info(f"{symbol}: direction_1d distribution: {distribution}")
    
    # Check FLAT class percentage
    flat_count = counts[vals == 0][0] if 0 in vals else 0
    flat_pct = (flat_count / total) * 100
    
    if flat_pct < 15.0:
        logger.warning(f"{symbol}: FLAT class only {flat_pct:.1f}% - consider wider neutral band")
    elif flat_pct > 50.0:
        logger.warning(f"{symbol}: FLAT class {flat_pct:.1f}% - consider narrower neutral band")

def validate_symbol_targets(df: pd.DataFrame, symbol: str) -> None:
    """
    Validate targets for a single symbol.
    
    Args:
        df: DataFrame with direction_1d column
        symbol: Symbol name for error reporting
        
    Raises:
        RuntimeError: If validation fails
    """
    if "direction_1d" not in df.columns:
        raise RuntimeError(f"direction_1d missing for symbol {symbol}")
    
    targets = df["direction_1d"].dropna()
    if len(targets) == 0:
        raise RuntimeError(f"No valid targets found for symbol {symbol}")
    
    valid_values = {-1, 0, 1}
    actual_values = set(targets.unique())
    invalid = actual_values - valid_values
    
    if invalid:
        raise RuntimeError(f"Invalid target values in {symbol}: {invalid}. Expected values: {valid_values}")
    
    # Log validation success
    logger.info(f"✅ Target validation passed for {symbol}: {len(targets):,} samples")
    
    # Check for missing classes (warning, not error for individual symbols)
    missing_classes = valid_values - actual_values
    if missing_classes:
        logger.warning(f"{symbol}: Missing target classes: {missing_classes}")

def validate_symbol_targets(df: pd.DataFrame, symbol: str) -> None:
    """
    Validate targets for a single symbol after target creation.
    
    Args:
        df: DataFrame with direction_1d column
        symbol: Symbol name for error reporting
        
    Raises:
        RuntimeError: If direction_1d is missing or contains invalid values
    """
    if "direction_1d" not in df.columns:
        raise RuntimeError(f"direction_1d missing for symbol {symbol}")
    
    targets = df["direction_1d"].dropna()
    if len(targets) == 0:
        raise RuntimeError(f"No valid targets found for symbol {symbol}")
    
    valid_values = {-1, 0, 1}
    actual_values = set(targets.unique())
    invalid = actual_values - valid_values
    
    if invalid:
        raise RuntimeError(f"Invalid target values in {symbol}: {invalid}. Expected only {valid_values}")
    
    # Log target distribution statistics
    vals, counts = np.unique(targets.values, return_counts=True)
    total = len(targets)
    
    distribution = {}
    for val, count in zip(vals, counts):
        pct = (count / total) * 100
        class_name = {-1: "DOWN", 0: "FLAT", 1: "UP"}.get(val, f"Class_{val}")
        distribution[class_name] = f"{count} ({pct:.1f}%)"
    
    logger.info(f"{symbol} validation passed - target distribution: {distribution}")
    
    # Check for missing classes (warning, not error for individual symbols)
    missing_classes = valid_values - actual_values
    if missing_classes:
        missing_names = [
            {-1: "DOWN", 0: "FLAT", 1: "UP"}.get(cls, f"Class_{cls}") 
            for cls in missing_classes
        ]
        logger.warning(f"{symbol}: Missing target classes: {missing_names}")

def validate_global_targets(df: pd.DataFrame, required_classes: set = None) -> None:
    """
    Validate targets across all symbols after concatenation.
    
    Args:
        df: Combined DataFrame with direction_1d column
        required_classes: Set of required classes (default: {-1, 0, 1})
    """
    if required_classes is None:
        required_classes = {-1, 0, 1}
    
    if "direction_1d" not in df.columns:
        raise RuntimeError("direction_1d missing after preprocessing — aborting.")
    
    targets = df["direction_1d"].dropna()
    if len(targets) == 0:
        raise RuntimeError("No valid targets found in combined dataset")
    
    unique_vals = set(targets.unique())
    
    # Check for invalid values
    if not unique_vals.issubset(required_classes):
        invalid_vals = unique_vals - required_classes
        raise RuntimeError(f"Invalid target values found: {invalid_vals}")
    
    # Check for missing classes
    missing_classes = required_classes - unique_vals
    if missing_classes:
        raise RuntimeError(f"Missing target classes in combined dataset: {missing_classes}")
    
    # Log global distribution
    vals, counts = np.unique(targets.values, return_counts=True)
    total = len(targets)
    
    logger.info("=== GLOBAL TARGET DISTRIBUTION ===")
    for val, count in zip(vals, counts):
        pct = (count / total) * 100
        class_name = {-1: "DOWN", 0: "FLAT", 1: "UP"}.get(val, f"Class_{val}")
        logger.info(f"{class_name} ({val}): {count:,} ({pct:.1f}%)")
    
    # Validate FLAT class percentage
    flat_count = counts[vals == 0][0] if 0 in vals else 0
    flat_pct = (flat_count / total) * 100
    
    if flat_pct < 10.0:
        raise RuntimeError(f"FLAT class critically low: {flat_pct:.1f}% - increase neutral band")
    elif flat_pct < 20.0:
        logger.warning(f"FLAT class low: {flat_pct:.1f}% - consider wider neutral band")
    
    logger.info(f"✅ Global target validation passed: {total:,} samples across {len(unique_vals)} classes")

def encode_targets(y_raw: np.ndarray) -> np.ndarray:
    """
    Encode -1,0,1 targets to 0,1,2 for neural network training.
    
    Args:
        y_raw: Raw targets in {-1, 0, 1} format
        
    Returns:
        Encoded targets in {0, 1, 2} format
        
    Raises:
        ValueError: If input targets are not in expected format
    """
    if not isinstance(y_raw, np.ndarray):
        y_raw = np.array(y_raw)
    
    # Validate input format
    unique_vals = set(np.unique(y_raw[~np.isnan(y_raw)]))
    expected_vals = {-1, 0, 1}
    
    if not unique_vals.issubset(expected_vals):
        invalid_vals = unique_vals - expected_vals
        raise ValueError(f"Expected targets in {{-1,0,1}}, got invalid values: {invalid_vals}")
    
    # Encode: -1 -> 0, 0 -> 1, 1 -> 2
    encoded = (y_raw + 1).clip(0, 2)
    
    logger.debug(f"Encoded targets: {len(y_raw)} samples, classes: {set(np.unique(encoded[~np.isnan(encoded)]))}")
    
    return encoded.astype(np.int64)

def get_optimal_neutral_band(df: pd.DataFrame, 
                           close_col: str = "close",
                           target_flat_pct: float = 25.0,
                           bands_to_test: list = None) -> float:
    """
    Find optimal neutral band to achieve target FLAT percentage.
    
    Args:
        df: DataFrame with price data
        close_col: Close price column name
        target_flat_pct: Target percentage for FLAT class
        bands_to_test: List of bands to test (default: [0.002, 0.003, 0.004, 0.005, 0.006])
        
    Returns:
        Optimal neutral band
    """
    if bands_to_test is None:
        bands_to_test = [0.002, 0.003, 0.004, 0.005, 0.006]  # 0.2% to 0.6%
    
    if close_col not in df.columns:
        raise ValueError(f"Missing close column '{close_col}'")
    
    best_band = bands_to_test[0]
    best_diff = float('inf')
    
    logger.info(f"Testing neutral bands to achieve ~{target_flat_pct:.1f}% FLAT class")
    
    for band in bands_to_test:
        # Calculate forward return
        fwd_ret = df[close_col].shift(-1) / df[close_col] - 1.0
        
        # Create labels
        y = np.where(fwd_ret > band, 1,
                    np.where(fwd_ret < -band, -1, 0))
        
        # Calculate FLAT percentage
        targets = pd.Series(y).dropna()
        flat_count = (targets == 0).sum()
        flat_pct = (flat_count / len(targets)) * 100
        
        # Check how close to target
        diff = abs(flat_pct - target_flat_pct)
        if diff < best_diff:
            best_diff = diff
            best_band = band
        
        logger.info(f"Band ±{band*100:.1f}%: FLAT = {flat_pct:.1f}% (diff: {diff:.1f}%)")
    
    logger.info(f"Optimal neutral band: ±{best_band*100:.1f}% (FLAT: {target_flat_pct - best_diff:.1f}%)")
    return best_band