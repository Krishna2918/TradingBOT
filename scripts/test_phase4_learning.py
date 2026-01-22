#!/usr/bin/env python3
"""Phase 4 Validation: Model Performance Learning + Performance Prediction"""

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
from src.ai.model_performance_learner import get_performance_learner
from src.ai.performance_prediction_system import get_prediction_system

def test_performance_learner():
    """Test model performance learning functionality."""
    learner = get_performance_learner()
    
    # Test performance recording
    test_conditions = {
        'regime': 'trending',
        'volatility_level': 'medium',
        'market_phase': 'mid_day',
        'sector': 'tech'
    }
    
    # Record multiple performance samples
    for i in range(15):
        success = learner.record_performance(
            model_name="test_model",
            accuracy=0.7 + (i % 3) * 0.1,  # Vary accuracy
            brier_score=0.5 + (i % 2) * 0.2,  # Vary brier score
            conditions=test_conditions,
            prediction=0.75 + i * 0.01,
            actual_outcome=0.8 + i * 0.005,
            confidence=0.8,
            execution_time=0.1 + i * 0.01,
            context={"test_run": i}
        )
        assert success, f"Failed to record performance sample {i}"
    
    print("[PASS] Performance recording works")
    
    # Test performance summary
    summary = learner.get_model_performance_summary("test_model")
    assert summary['total_predictions'] == 15, "Incorrect prediction count"
    assert summary['avg_accuracy'] > 0.0, "No average accuracy calculated"
    assert summary['avg_brier_score'] > 0.0, "No average brier score calculated"
    assert 'performance_trend' in summary, "Missing performance trend"
    print("[PASS] Performance summary works")
    
    # Test performance prediction
    prediction = learner.predict_model_performance("test_model", test_conditions)
    assert prediction['model_name'] == "test_model", "Incorrect model name in prediction"
    assert prediction['predicted_accuracy'] > 0.0, "No accuracy prediction"
    assert prediction['predicted_brier_score'] > 0.0, "No brier score prediction"
    assert prediction['sample_size'] > 0, "No similar samples found"
    print("[PASS] Performance prediction works")
    
    # Test learning insights
    insights = learner.get_learning_insights("test_model")
    # Insights may be empty if not enough data for learning
    assert isinstance(insights, list), "Insights not returned as list"
    print("[PASS] Learning insights work")
    
    # Test learning statistics
    stats = learner.get_learning_statistics()
    assert stats['total_records'] == 15, "Incorrect total records"
    assert stats['unique_models'] == 1, "Incorrect unique model count"
    assert 'avg_accuracy' in stats, "Missing average accuracy in stats"
    print("[PASS] Learning statistics work")
    
    return True

def test_performance_prediction_system():
    """Test performance prediction system functionality."""
    predictor = get_prediction_system()
    
    # Test adding training data
    test_conditions = {
        'regime': 'trending',
        'volatility_level': 'medium',
        'market_phase': 'mid_day',
        'sector': 'tech',
        'volatility_zscore': 1.2,
        'correlation': 0.7,
        'liquidity_score': 0.8,
        'news_sentiment': 0.3
    }
    
    # Add multiple training samples (need at least 100 for training)
    for i in range(120):
        success = predictor.add_training_data(
            model_name="test_model",
            conditions=test_conditions,
            accuracy=0.6 + (i % 5) * 0.05,  # Vary accuracy
            brier_score=0.4 + (i % 3) * 0.1,  # Vary brier score
            execution_time=0.1 + (i % 4) * 0.02  # Vary execution time
        )
        assert success, f"Failed to add training data sample {i}"
    
    print("[PASS] Training data addition works")
    
    # Test model training
    training_success = predictor.train_models()
    assert training_success, "Model training failed"
    assert predictor.is_trained, "System not marked as trained"
    print("[PASS] Model training works")
    
    # Test performance prediction
    prediction = predictor.predict_performance("test_model", test_conditions)
    assert prediction.model_name == "test_model", "Incorrect model name"
    assert prediction.predicted_accuracy > 0.0, "No accuracy prediction"
    assert prediction.predicted_brier_score > 0.0, "No brier score prediction"
    assert prediction.predicted_execution_time > 0.0, "No execution time prediction"
    assert prediction.prediction_confidence >= 0.0, "Invalid prediction confidence"
    assert len(prediction.confidence_interval) == 2, "Invalid confidence interval"
    print("[PASS] Performance prediction works")
    
    # Test prediction statistics
    stats = predictor.get_prediction_statistics()
    assert stats['is_trained'] == True, "System not marked as trained in stats"
    assert stats['training_samples'] == 120, "Incorrect training sample count"
    assert stats['trained_models'] > 0, "No trained models"
    assert 'model_types' in stats, "Missing model types in stats"
    assert 'model_metrics' in stats, "Missing model metrics in stats"
    print("[PASS] Prediction statistics work")
    
    return True

def test_integration():
    """Test integration between performance learner and prediction system."""
    learner = get_performance_learner()
    predictor = get_prediction_system()
    
    # Test conditions
    test_conditions = {
        'regime': 'ranging',
        'volatility_level': 'low',
        'market_phase': 'open',
        'sector': 'finance',
        'volatility_zscore': 0.5,
        'correlation': 0.6,
        'liquidity_score': 0.9,
        'news_sentiment': 0.1
    }
    
    # Record performance in learner
    learner_success = learner.record_performance(
        model_name="integration_model",
        accuracy=0.85,
        brier_score=0.3,
        conditions=test_conditions,
        prediction=0.8,
        actual_outcome=0.82,
        confidence=0.9,
        execution_time=0.05
    )
    assert learner_success, "Failed to record performance in learner"
    
    # Add training data to predictor
    predictor_success = predictor.add_training_data(
        model_name="integration_model",
        conditions=test_conditions,
        accuracy=0.85,
        brier_score=0.3,
        execution_time=0.05
    )
    assert predictor_success, "Failed to add training data to predictor"
    
    # Test that both systems can work with the same model
    learner_summary = learner.get_model_performance_summary("integration_model")
    assert learner_summary['total_predictions'] >= 1, "Learner doesn't have model data"
    
    # Train predictor if not already trained
    if not predictor.is_trained:
        predictor.train_models()
    
    predictor_prediction = predictor.predict_performance("integration_model", test_conditions)
    assert predictor_prediction.model_name == "integration_model", "Predictor prediction mismatch"
    
    print("[PASS] Integration between systems works")
    
    return True

def test_ml_functionality():
    """Test ML-specific functionality."""
    predictor = get_prediction_system()
    
    # Add more diverse training data for better ML testing
    conditions_variants = [
        {'regime': 'trending', 'volatility_level': 'high', 'sector': 'tech'},
        {'regime': 'ranging', 'volatility_level': 'low', 'sector': 'finance'},
        {'regime': 'volatile', 'volatility_level': 'medium', 'sector': 'healthcare'},
        {'regime': 'calm', 'volatility_level': 'low', 'sector': 'energy'}
    ]
    
    # Add training data for each condition variant
    for i, conditions in enumerate(conditions_variants):
        for j in range(5):  # 5 samples per condition
            predictor.add_training_data(
                model_name=f"ml_test_model_{i}",
                conditions=conditions,
                accuracy=0.6 + i * 0.05 + j * 0.01,
                brier_score=0.5 - i * 0.05 + j * 0.02,
                execution_time=0.1 + i * 0.01 + j * 0.005
            )
    
    # Train models
    training_success = predictor.train_models()
    assert training_success, "ML model training failed"
    
    # Test predictions on different conditions
    for conditions in conditions_variants:
        prediction = predictor.predict_performance("ml_test_model_0", conditions)
        assert prediction.predicted_accuracy > 0.0, f"No accuracy prediction for conditions {conditions}"
        assert prediction.predicted_brier_score > 0.0, f"No brier score prediction for conditions {conditions}"
        assert prediction.predicted_execution_time > 0.0, f"No execution time prediction for conditions {conditions}"
    
    print("[PASS] ML functionality works")
    
    return True

if __name__ == "__main__":
    try:
        test_performance_learner()
        test_performance_prediction_system()
        test_integration()
        test_ml_functionality()
        print("\n[PASS] PHASE 4 VALIDATION: PASSED")
        exit(0)
    except AssertionError as e:
        print(f"\n[FAIL] PHASE 4 VALIDATION: FAILED - {e}")
        exit(1)
    except Exception as e:
        print(f"\n[ERROR] PHASE 4 VALIDATION: ERROR - {e}")
        exit(1)