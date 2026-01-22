#!/usr/bin/env python3
"""Phase 6 Validation: Advanced Ensemble Methods & Model Evolution"""

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
from src.ai.advanced_ensemble_methods import AdvancedEnsembleMethods
from src.ai.model_evolution_system import ModelEvolutionSystem, EvolutionTrigger

def test_advanced_ensemble():
    """Test advanced ensemble methods."""
    ensemble = AdvancedEnsembleMethods()
    
    # Test dynamic stacking
    predictions = {
        'model_a': 0.75,
        'model_b': 0.73,
        'model_c': 0.77
    }
    confidences = {
        'model_a': 0.8,
        'model_b': 0.75,
        'model_c': 0.85
    }
    market_context = {
        'volatility': 0.02,
        'trend_strength': 0.3,
        'regime': 'trend'
    }
    
    result = ensemble.dynamic_stacking_ensemble(predictions, confidences, market_context)
    assert result is not None, "Dynamic stacking failed"
    assert result.ensemble_method == 'dynamic_stacking', "Wrong ensemble method"
    assert result.prediction is not None, "No prediction generated"
    assert result.confidence > 0, "No confidence calculated"
    print(f"[PASS] Dynamic stacking: prediction={result.prediction:.3f}, confidence={result.confidence:.3f}")
    
    # Test hierarchical ensemble
    model_categories = {
        'model_a': 'trend',
        'model_b': 'trend',
        'model_c': 'mean_reversion'
    }
    
    result_hier = ensemble.hierarchical_ensemble(predictions, confidences, model_categories)
    assert result_hier is not None, "Hierarchical ensemble failed"
    assert result_hier.ensemble_method == 'hierarchical', "Wrong ensemble method"
    assert result_hier.prediction is not None, "No prediction generated"
    print(f"[PASS] Hierarchical ensemble: prediction={result_hier.prediction:.3f}")
    
    # Test adaptive weighting
    recent_performance = {
        'model_a': 0.8,
        'model_b': 0.7,
        'model_c': 0.9
    }
    
    result_adapt = ensemble.adaptive_weighting_ensemble(predictions, confidences, recent_performance)
    assert result_adapt is not None, "Adaptive weighting failed"
    assert result_adapt.ensemble_method == 'adaptive_weighting', "Wrong ensemble method"
    assert result_adapt.prediction is not None, "No prediction generated"
    print(f"[PASS] Adaptive weighting: prediction={result_adapt.prediction:.3f}")
    
    return True

def test_model_evolution():
    """Test model evolution system."""
    evolution = ModelEvolutionSystem(retrain_threshold=0.05)
    
    # Create sample training data
    dates = pd.date_range(start='2024-01-01', periods=100)
    training_data = pd.DataFrame({
        'date': dates,
        'open': np.random.uniform(100, 105, 100),
        'high': np.random.uniform(105, 110, 100),
        'low': np.random.uniform(95, 100, 100),
        'close': np.random.uniform(100, 105, 100),
        'volume': np.random.uniform(1000000, 2000000, 100)
    })
    
    validation_data = training_data.tail(20).copy()
    
    # Test evolution triggers
    triggers = evolution.check_evolution_triggers('test_model', 0.6, training_data)
    assert isinstance(triggers, list), "Triggers should be a list"
    print(f"[PASS] Evolution triggers: {[t.value for t in triggers]}")
    
    # Test model evolution
    result = evolution.evolve_model('test_model', triggers, training_data, validation_data)
    assert result is not None, "Evolution result is None"
    assert result.model_name == 'test_model', "Wrong model name"
    assert result.timestamp is not None, "No timestamp"
    assert isinstance(result.success, bool), "Success should be boolean"
    print(f"[PASS] Model evolution: success={result.success}, improvement={result.improvement:.3f}")
    
    # Test evolution statistics
    stats = evolution.get_evolution_statistics()
    assert 'total_evolutions' in stats, "Missing total_evolutions"
    assert stats['total_evolutions'] >= 1, "Should have at least one evolution"
    print(f"[PASS] Evolution statistics: {stats['total_evolutions']} evolutions")
    
    return True

if __name__ == "__main__":
    try:
        test_advanced_ensemble()
        test_model_evolution()
        print("\n[PASS] PHASE 6 VALIDATION: PASSED")
        exit(0)
    except AssertionError as e:
        print(f"\n[FAIL] PHASE 6 VALIDATION: FAILED - {e}")
        exit(1)
