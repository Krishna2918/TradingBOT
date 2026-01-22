#!/usr/bin/env python3
"""Phase 6 Validation: Advanced Ensemble Methods + Model Evolution"""

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

from datetime import datetime, timedelta
from src.ai.advanced_ensemble_methods import get_ensemble_manager
from src.ai.model_evolution_system import get_evolution_engine, ModelPerformance, EvolutionStrategy

def test_dynamic_stacking_ensemble():
    """Test dynamic stacking ensemble functionality."""
    ensemble_manager = get_ensemble_manager()
    
    # Test adding training samples
    for i in range(20):
        predictions = {
            'model_a': 0.7 + i * 0.01,
            'model_b': 0.65 + i * 0.015,
            'model_c': 0.75 + i * 0.005
        }
        actual_outcome = 0.7 + i * 0.01 + (i % 3 - 1) * 0.02  # Add some variation
        market_conditions = {
            'regime': 'trending' if i % 2 == 0 else 'ranging',
            'volatility_regime': 'medium',
            'liquidity_regime': 'high'
        }
        
        ensemble_manager.add_training_sample(predictions, actual_outcome, market_conditions)
    
    print("[PASS] Training sample addition works")
    
    # Test dynamic stacking prediction
    test_predictions = {
        'model_a': 0.8,
        'model_b': 0.75,
        'model_c': 0.82
    }
    test_conditions = {
        'regime': 'trending',
        'volatility_regime': 'medium',
        'liquidity_regime': 'high'
    }
    
    result = ensemble_manager.predict_ensemble(test_predictions, test_conditions, 'dynamic_stacking')
    assert result is not None, "No prediction result from dynamic stacking"
    assert result.final_prediction > 0.0, "Invalid final prediction"
    assert result.confidence > 0.0, "Invalid confidence"
    assert result.method_used == 'dynamic_stacking', "Wrong method used"
    assert len(result.weights) > 0, "No weights calculated"
    print("[PASS] Dynamic stacking prediction works")
    
    return True

def test_hierarchical_ensemble():
    """Test hierarchical ensemble functionality."""
    ensemble_manager = get_ensemble_manager()
    
    # Test hierarchical prediction with categorized models
    test_predictions = {
        'rsi_model': 0.7,  # Technical
        'macd_model': 0.75,  # Technical
        'pe_ratio_model': 0.6,  # Fundamental
        'earnings_model': 0.65,  # Fundamental
        'news_sentiment_model': 0.8,  # Sentiment
        'social_sentiment_model': 0.78,  # Sentiment
        'trend_following_model': 0.85,  # Trend
        'momentum_model': 0.82,  # Trend
        'mean_reversion_model': 0.4,  # Mean reversion
        'oscillator_model': 0.45,  # Mean reversion
        'volatility_model': 0.6,  # Volatility
        'garch_model': 0.55  # Volatility
    }
    
    test_conditions = {
        'regime': 'trending',
        'volatility_regime': 'medium'
    }
    
    result = ensemble_manager.predict_ensemble(test_predictions, test_conditions, 'hierarchical')
    assert result is not None, "No prediction result from hierarchical ensemble"
    assert result.final_prediction > 0.0, "Invalid final prediction"
    assert result.confidence > 0.0, "Invalid confidence"
    assert result.method_used == 'hierarchical', "Wrong method used"
    assert len(result.weights) > 0, "No weights calculated"
    print("[PASS] Hierarchical ensemble prediction works")
    
    return True

def test_adaptive_weighting_ensemble():
    """Test adaptive weighting ensemble functionality."""
    ensemble_manager = get_ensemble_manager()
    
    # Test performance updates
    for i in range(15):
        # Simulate different model performances
        model_a_pred = 0.7 + i * 0.01
        model_b_pred = 0.65 + i * 0.015
        model_c_pred = 0.75 + i * 0.005
        
        actual_outcome = 0.7 + i * 0.01
        
        # Update performance for each model
        ensemble_manager.update_performance('model_a', model_a_pred, actual_outcome)
        ensemble_manager.update_performance('model_b', model_b_pred, actual_outcome)
        ensemble_manager.update_performance('model_c', model_c_pred, actual_outcome)
    
    print("[PASS] Performance updates work")
    
    # Test adaptive weighting prediction
    test_predictions = {
        'model_a': 0.8,
        'model_b': 0.75,
        'model_c': 0.82
    }
    
    test_conditions = {
        'regime': 'ranging',
        'volatility_regime': 'low'
    }
    
    result = ensemble_manager.predict_ensemble(test_predictions, test_conditions, 'adaptive_weighting')
    assert result is not None, "No prediction result from adaptive weighting"
    assert result.final_prediction > 0.0, "Invalid final prediction"
    assert result.confidence > 0.0, "Invalid confidence"
    assert result.method_used == 'adaptive_weighting', "Wrong method used"
    assert len(result.weights) > 0, "No weights calculated"
    print("[PASS] Adaptive weighting prediction works")
    
    return True

def test_combined_ensemble():
    """Test combined ensemble prediction."""
    ensemble_manager = get_ensemble_manager()
    
    test_predictions = {
        'model_a': 0.75,
        'model_b': 0.73,
        'model_c': 0.77
    }
    
    test_conditions = {
        'regime': 'volatile',
        'volatility_regime': 'high',
        'liquidity_regime': 'medium'
    }
    
    result = ensemble_manager.predict_ensemble(test_predictions, test_conditions, 'combined')
    assert result is not None, "No prediction result from combined ensemble"
    assert result.final_prediction > 0.0, "Invalid final prediction"
    assert result.confidence > 0.0, "Invalid confidence"
    assert result.method_used == 'combined', "Wrong method used"
    assert len(result.weights) > 0, "No weights calculated"
    print("[PASS] Combined ensemble prediction works")
    
    # Test ensemble statistics
    stats = ensemble_manager.get_ensemble_statistics()
    assert 'total_ensembles' in stats, "Missing total_ensembles in stats"
    assert 'ensemble_methods' in stats, "Missing ensemble_methods in stats"
    assert 'ensemble_weights' in stats, "Missing ensemble_weights in stats"
    assert stats['total_ensembles'] >= 3, "Insufficient ensemble count"
    print("[PASS] Ensemble statistics work")
    
    return True

def test_model_evolution_engine():
    """Test model evolution engine functionality."""
    evolution_engine = get_evolution_engine()
    
    # Test model registration
    def dummy_evolution_callback(evolution_event):
        return f"Evolution executed for {evolution_event.model_name}"
    
    evolution_engine.register_model('test_model_1', 'dummy_instance', dummy_evolution_callback)
    evolution_engine.register_model('test_model_2', 'dummy_instance', dummy_evolution_callback)
    
    assert 'test_model_1' in evolution_engine.model_registry, "Model 1 not registered"
    assert 'test_model_2' in evolution_engine.model_registry, "Model 2 not registered"
    print("[PASS] Model registration works")
    
    # Test performance updates
    for i in range(25):
        performance = ModelPerformance(
            model_name='test_model_1',
            accuracy=0.8 - i * 0.01,  # Declining performance
            precision=0.75 - i * 0.008,
            recall=0.85 - i * 0.012,
            f1_score=0.8 - i * 0.01,
            sharpe_ratio=1.5 - i * 0.05,
            max_drawdown=0.1 + i * 0.005,
            win_rate=0.6 - i * 0.008,
            profit_factor=1.2 - i * 0.01,
            timestamp=datetime.now() - timedelta(hours=25-i),
            market_conditions={
                'regime': 'trending' if i % 2 == 0 else 'ranging',
                'volatility_regime': 'medium'
            }
        )
        
        evolution_engine.update_performance('test_model_1', performance)
    
    print("[PASS] Performance updates work")
    
    # Test evolution statistics
    stats = evolution_engine.get_evolution_statistics()
    assert 'total_models_registered' in stats, "Missing total_models_registered in stats"
    assert 'total_evolution_events' in stats, "Missing total_evolution_events in stats"
    assert 'evolution_by_trigger' in stats, "Missing evolution_by_trigger in stats"
    assert 'evolution_by_strategy' in stats, "Missing evolution_by_strategy in stats"
    assert stats['total_models_registered'] >= 2, "Insufficient registered models"
    print("[PASS] Evolution statistics work")
    
    # Test model evolution history
    history = evolution_engine.get_model_evolution_history('test_model_1')
    assert isinstance(history, list), "Evolution history not returned as list"
    print("[PASS] Model evolution history works")
    
    return True

def test_manual_evolution_trigger():
    """Test manual evolution triggering."""
    evolution_engine = get_evolution_engine()
    
    # Test manual evolution trigger
    success = evolution_engine.trigger_manual_evolution(
        'test_model_1',
        EvolutionStrategy.HYPERPARAMETER_TUNING,
        "Manual test evolution"
    )
    
    assert success, "Manual evolution trigger failed"
    print("[PASS] Manual evolution trigger works")
    
    # Test evolution with non-registered model
    success = evolution_engine.trigger_manual_evolution(
        'non_existent_model',
        EvolutionStrategy.FULL_RETRAIN,
        "Test with non-existent model"
    )
    
    assert not success, "Manual evolution should fail for non-registered model"
    print("[PASS] Manual evolution validation works")
    
    return True

def test_integration():
    """Test integration between ensemble methods and evolution system."""
    ensemble_manager = get_ensemble_manager()
    evolution_engine = get_evolution_engine()
    
    # Create a model that will be tracked by both systems
    model_name = 'integrated_model'
    
    # Register model for evolution
    def evolution_callback(evolution_event):
        # When evolution is triggered, update ensemble weights
        if evolution_event.strategy == EvolutionStrategy.ENSEMBLE_ADJUSTMENT:
            # Simulate ensemble adjustment
            return "Ensemble weights adjusted"
        return "Evolution completed"
    
    evolution_engine.register_model(model_name, 'dummy_instance', evolution_callback)
    
    # Simulate performance decline to trigger evolution
    for i in range(30):
        performance = ModelPerformance(
            model_name=model_name,
            accuracy=0.9 - i * 0.02,  # Significant decline
            precision=0.85 - i * 0.015,
            recall=0.9 - i * 0.02,
            f1_score=0.87 - i * 0.018,
            sharpe_ratio=2.0 - i * 0.1,
            max_drawdown=0.05 + i * 0.01,
            win_rate=0.7 - i * 0.015,
            profit_factor=1.5 - i * 0.05,
            timestamp=datetime.now() - timedelta(hours=30-i),
            market_conditions={
                'regime': 'trending',
                'volatility_regime': 'medium'
            }
        )
        
        evolution_engine.update_performance(model_name, performance)
    
    # Test ensemble prediction with evolved model
    test_predictions = {
        model_name: 0.75,
        'other_model': 0.73
    }
    
    test_conditions = {
        'regime': 'trending',
        'volatility_regime': 'medium'
    }
    
    result = ensemble_manager.predict_ensemble(test_predictions, test_conditions, 'combined')
    assert result is not None, "No prediction result from integrated system"
    assert result.final_prediction > 0.0, "Invalid final prediction"
    
    # Check that evolution occurred
    stats = evolution_engine.get_evolution_statistics()
    assert stats['total_evolution_events'] > 0, "No evolution events occurred"
    
    print("[PASS] Integration between systems works")
    
    return True

if __name__ == "__main__":
    try:
        test_dynamic_stacking_ensemble()
        test_hierarchical_ensemble()
        test_adaptive_weighting_ensemble()
        test_combined_ensemble()
        test_model_evolution_engine()
        test_manual_evolution_trigger()
        test_integration()
        print("\n[PASS] PHASE 6 VALIDATION: PASSED")
        exit(0)
    except AssertionError as e:
        print(f"\n[FAIL] PHASE 6 VALIDATION: FAILED - {e}")
        exit(1)
    except Exception as e:
        print(f"\n[ERROR] PHASE 6 VALIDATION: ERROR - {e}")
        exit(1)
