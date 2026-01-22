#!/usr/bin/env python3
"""Debug Phase 7 Feature Engineering Issues"""

import sys
import os
import io

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except AttributeError:
        pass  # Already wrapped or not needed

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.ai.advanced_feature_engineering import get_feature_pipeline

def debug_feature_engineering():
    """Debug feature engineering issues."""
    pipeline = get_feature_pipeline()
    
    # Create simple market data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
    
    base_price = 100.0
    prices = [base_price]
    volumes = [1000000]
    
    for i in range(99):
        price_change = np.random.normal(0.001, 0.02)
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
        
        volume_change = np.random.normal(0, 0.1)
        new_volume = volumes[-1] * (1 + volume_change)
        volumes.append(max(100000, new_volume))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {data.columns.tolist()}")
    
    # Test feature engineering
    feature_set = pipeline.engineer_features(data)
    
    print(f"Feature set: {feature_set}")
    print(f"Features count: {len(feature_set.features)}")
    print(f"Feature names count: {len(feature_set.feature_names)}")
    print(f"Feature names: {feature_set.feature_names[:10]}...")  # First 10
    
    # Test feature importance
    importance = pipeline.get_feature_importance(feature_set)
    print(f"Importance count: {len(importance)}")
    print(f"Importance keys: {list(importance.keys())[:10]}...")  # First 10
    
    # Check for mismatch
    if len(importance) != len(feature_set.feature_names):
        print(f"MISMATCH: importance={len(importance)}, feature_names={len(feature_set.feature_names)}")
        
        # Find missing features
        importance_keys = set(importance.keys())
        feature_names_set = set(feature_set.feature_names)
        
        missing_in_importance = feature_names_set - importance_keys
        missing_in_features = importance_keys - feature_names_set
        
        if missing_in_importance:
            print(f"Missing in importance: {missing_in_importance}")
        if missing_in_features:
            print(f"Missing in features: {missing_in_features}")
    else:
        print("No mismatch found")

if __name__ == "__main__":
    debug_feature_engineering()
