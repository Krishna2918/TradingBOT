#!/usr/bin/env python3
"""Phase 2 Validation: Advanced Models Integration"""

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
from src.ai.advanced_models.model_integration import AdvancedModelIntegration
from src.ai.advanced_models.performance_optimizer import PerformanceOptimizer
from src.ai.advanced_models.feature_pipeline import AdvancedFeaturePipeline

def test_model_integration():
    """Test model integration fixes."""
    integration = AdvancedModelIntegration(
        enable_deep_learning=False,
        enable_time_series=False,
        enable_reinforcement_learning=False,
        enable_nlp=False
    )
    
    sample_data = {
        'market_data': pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })
    }
    
    # Test predict_sync has timestamp
    result = integration.predict_sync(sample_data)
    assert 'timestamp' in result, "Missing timestamp in predict_sync"
    print("[PASS] predict_sync has timestamp")
    
    # Test ensemble_predict structure
    ensemble_result = integration.ensemble_predict(sample_data)
    assert 'ensemble_method' in ensemble_result, "Missing ensemble_method"
    assert 'individual_predictions' in ensemble_result, "Missing individual_predictions"
    assert 'timestamp' in ensemble_result, "Missing timestamp in ensemble"
    print("[PASS] ensemble_predict has correct structure")
    
    return True

def test_performance_optimizer():
    """Test performance optimizer fixes."""
    optimizer = PerformanceOptimizer()
    
    result = optimizer.optimize_system()
    assert 'total_execution_time' in result, "Missing total_execution_time"
    assert 'memory_cleanups' in result, "Missing memory_cleanups"
    print("[PASS] Performance optimizer has all required keys")
    
    return True

def test_feature_optimizer():
    """Test feature optimizer fixes."""
    pipeline = AdvancedFeaturePipeline()
    
    features = pd.DataFrame({
        'f1': [1, 2, 3, 4, 5],
        'f2': [1.1, 2.1, 3.1, 4.1, 5.1],  # Highly correlated with f1
        'f3': [10, 20, 30, 40, 50]
    })
    
    result = pipeline.feature_optimizer.optimize_features(features)
    assert 'optimization_strategies' in result, "Missing optimization_strategies"
    print("[PASS] Feature optimizer has optimization_strategies")
    
    return True

if __name__ == "__main__":
    try:
        test_model_integration()
        test_performance_optimizer()
        test_feature_optimizer()
        print("\n[PASS] PHASE 2 VALIDATION: PASSED")
        exit(0)
    except AssertionError as e:
        print(f"\n[FAIL] PHASE 2 VALIDATION: FAILED - {e}")
        exit(1)
