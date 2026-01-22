#!/usr/bin/env python3
"""
Phase 5 Simple Test - Adaptive Ensemble Weights
==============================================

Simplified test focusing on core Phase 5 functionality without complex dependencies.
"""

import sys
import os
import time
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_adaptive_weight_manager():
    """Test adaptive weight manager basic functionality."""
    print("Testing Adaptive Weight Manager...")
    
    try:
        from ai.adaptive_weights import (
            AdaptiveWeightManager, get_adaptive_weight_manager,
            add_model_prediction, get_ensemble_weights, update_ensemble_weights
        )
        
        # Test initialization
        manager = AdaptiveWeightManager(window_size_days=7, min_predictions=5)
        print("[OK] Adaptive weight manager initialized")
        
        # Test adding predictions
        base_date = datetime.now() - timedelta(days=3)
        
        # Add predictions for model A (good performance)
        for i in range(10):
            add_model_prediction(
                model_name="technical_analyst",
                symbol="TEST",
                predicted_probability=0.8,
                actual_outcome="WIN",
                prediction_date=base_date + timedelta(hours=i)
            )
        
        # Add predictions for model B (poor performance)
        for i in range(10):
            add_model_prediction(
                model_name="sentiment_analyst",
                symbol="TEST",
                predicted_probability=0.8,
                actual_outcome="LOSS",
                prediction_date=base_date + timedelta(hours=i)
            )
        
        print("[OK] Model predictions added")
        
        # Test weight calculation
        ensemble_weights = update_ensemble_weights()
        print(f"[OK] Ensemble weights calculated: {ensemble_weights.weights}")
        
        # Test getting weights
        weights = get_ensemble_weights()
        print(f"[OK] Retrieved ensemble weights: {weights}")
        
        # Verify weights sum to 1
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01, f"Weights should sum to 1, got {total_weight}"
        print("[OK] Weights sum to 1.0")
        
        # Verify better model has higher weight
        if "technical_analyst" in weights and "sentiment_analyst" in weights:
            assert weights["technical_analyst"] > weights["sentiment_analyst"], \
                "Better performing model should have higher weight"
            print("[OK] Better performing model has higher weight")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Adaptive weight manager test failed: {e}")
        return False

def test_weight_calculation_logic():
    """Test weight calculation and tie-break logic."""
    print("\nTesting Weight Calculation Logic...")
    
    try:
        from ai.adaptive_weights import AdaptiveWeightManager
        
        manager = AdaptiveWeightManager()
        
        # Test Brier score calculation
        perfect_predictions = [
            {"predicted_probability": 0.9, "actual_outcome": "WIN"},
            {"predicted_probability": 0.1, "actual_outcome": "LOSS"},
            {"predicted_probability": 0.8, "actual_outcome": "WIN"},
            {"predicted_probability": 0.2, "actual_outcome": "LOSS"}
        ]
        
        brier_score = manager._calculate_brier_score(perfect_predictions)
        assert brier_score < 0.1, f"Perfect predictions should have low Brier score, got {brier_score}"
        print("[OK] Brier score calculation working")
        
        # Test accuracy calculation
        accuracy = manager._calculate_accuracy(perfect_predictions)
        assert accuracy == 1.0, f"Perfect predictions should have 100% accuracy, got {accuracy}"
        print("[OK] Accuracy calculation working")
        
        # Test weight entropy calculation
        uniform_weights = {"model_a": 0.5, "model_b": 0.5}
        concentrated_weights = {"model_a": 0.9, "model_b": 0.1}
        
        entropy_uniform = manager._calculate_weight_entropy(uniform_weights)
        entropy_concentrated = manager._calculate_weight_entropy(concentrated_weights)
        
        assert entropy_uniform > entropy_concentrated, \
            "Uniform weights should have higher entropy than concentrated weights"
        print("[OK] Weight entropy calculation working")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Weight calculation logic test failed: {e}")
        return False

def test_export_import_functionality():
    """Test export/import functionality."""
    print("\nTesting Export/Import Functionality...")
    
    try:
        from ai.adaptive_weights import AdaptiveWeightManager, add_model_prediction
        
        # Create manager and add some data
        manager = AdaptiveWeightManager()
        
        base_date = datetime.now() - timedelta(days=3)
        manager.add_prediction(
            model_name="test_model",
            symbol="TEST",
            predicted_probability=0.8,
            actual_outcome="WIN",
            prediction_date=base_date
        )
        
        # Export data
        exported_data = manager.export_weights_data()
        print("[OK] Data exported successfully")
        
        # Check exported data structure
        assert "model_predictions" in exported_data, "Should include model predictions"
        assert "model_performance" in exported_data, "Should include model performance"
        assert "ensemble_weights" in exported_data, "Should include ensemble weights"
        print("[OK] Exported data has correct structure")
        
        # Create new manager and import data
        new_manager = AdaptiveWeightManager()
        new_manager.import_weights_data(exported_data)
        print("[OK] Data imported successfully")
        
        # Verify data was imported
        assert len(new_manager.model_predictions) >= 1, f"Should have at least 1 prediction, got {len(new_manager.model_predictions)}"
        # Check if any prediction has the correct model name
        model_names = [p["model_name"] for p in new_manager.model_predictions]
        assert "test_model" in model_names, f"Should have test_model prediction, got {model_names}"
        print("[OK] Imported data is correct")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Export/import functionality test failed: {e}")
        return False

def test_database_integration():
    """Test database integration for model performance."""
    print("\nTesting Database Integration...")
    
    try:
        from config.database import (
            get_database_manager, log_model_performance,
            get_model_performance_history, get_latest_model_performance
        )
        
        db_manager = get_database_manager()
        print("[OK] Database manager initialized")
        
        # Test logging model performance
        timestamp = datetime.now()
        window_start = timestamp - timedelta(days=7)
        window_end = timestamp
        
        result = log_model_performance(
            timestamp=timestamp,
            model="test_model",
            brier_score=0.25,
            accuracy=0.75,
            n_predictions=20,
            weight=0.4,
            window_start=window_start,
            window_end=window_end,
            mode="DEMO"
        )
        print(f"[OK] Logged model performance: {result}")
        
        # Test retrieving performance history
        history = get_model_performance_history(model="test_model", mode="DEMO")
        print(f"[OK] Retrieved performance history: {len(history)} records")
        
        # Test retrieving latest performance
        latest = get_latest_model_performance(model="test_model", mode="DEMO")
        print(f"[OK] Retrieved latest performance: {len(latest)} records")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Database integration test failed: {e}")
        return False

def main():
    """Run all Phase 5 simple tests."""
    print("=" * 60)
    print("PHASE 5 SIMPLE TEST - ADAPTIVE ENSEMBLE WEIGHTS")
    print("=" * 60)
    
    start_time = time.time()
    
    tests = [
        ("Adaptive Weight Manager", test_adaptive_weight_manager),
        ("Weight Calculation Logic", test_weight_calculation_logic),
        ("Export/Import Functionality", test_export_import_functionality),
        ("Database Integration", test_database_integration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[FAIL] {test_name} test failed with exception: {e}")
            failed += 1
    
    duration = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("PHASE 5 SIMPLE TEST SUMMARY")
    print("=" * 60)
    print(f"Duration: {duration:.1f}s")
    print(f"Tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if failed == 0:
        print("\n[OK] ALL TESTS PASSED - Phase 5 core functionality working correctly!")
        print("[READY] Ready for Phase 6: Drawdown-Aware Kelly & ATR Brackets!")
    else:
        print(f"\n[FAIL] {failed} TESTS FAILED - Check implementation")
    
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
