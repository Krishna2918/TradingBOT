#!/usr/bin/env python3
"""
Phase 5 Smoke Test - Adaptive Ensemble Weights
==============================================

Quick validation of Phase 5 implementation:
- Adaptive weight manager functionality
- Ensemble integration with adaptive weights
- Multi-model integration
- Database integration
- Weight calculation and tie-break logic
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
        from src.ai.adaptive_weights import (
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

def test_enhanced_ensemble_integration():
    """Test enhanced ensemble integration with adaptive weights."""
    print("\nTesting Enhanced Ensemble Integration...")
    
    try:
        from src.ai.enhanced_ensemble import EnhancedEnsemble, get_enhanced_ensemble
        
        ensemble = get_enhanced_ensemble()
        print("[OK] Enhanced ensemble initialized")
        
        # Test market analysis with adaptive weights
        market_data = {
            "current_price": 150.0,
            "rsi": 45.0,
            "macd": 0.5,
            "sma_20": 148.0,
            "sma_50": 145.0,
            "bollinger_position": 0.3,
            "volume_ratio": 1.2,
            "atr": 0.02,
            "sentiment_score": 0.6,
            "fundamental_score": 0.7,
            "market_regime": "BULL",
            "sector_performance": 0.1,
            "news_impact": 0.3,
            "volatility": 0.025,
            "volume_trend": "HIGH"
        }
        
        # Build market analysis
        market_analysis = ensemble._build_market_analysis("TEST", market_data)
        print("[OK] Market analysis built")
        
        # Run ensemble analysis
        result = ensemble._run_ensemble_analysis(market_analysis, "ENTRY")
        print("[OK] Ensemble analysis completed")
        
        # Check that result includes weights
        assert "weights" in result, "Result should include weights"
        assert "technical" in result["weights"], "Should include technical weight"
        assert "sentiment" in result["weights"], "Should include sentiment weight"
        assert "fundamental" in result["weights"], "Should include fundamental weight"
        assert "risk" in result["weights"], "Should include risk weight"
        assert "regime" in result["weights"], "Should include regime weight"
        print("[OK] Result includes all weight components")
        
        # Check weights sum to approximately 1
        total_weight = sum(result["weights"].values())
        assert abs(total_weight - 1.0) < 0.01, f"Weights should sum to 1, got {total_weight}"
        print("[OK] Ensemble weights sum to 1.0")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Enhanced ensemble integration test failed: {e}")
        return False

def test_multi_model_integration():
    """Test multi-model integration with adaptive weights."""
    print("\nTesting Multi-Model Integration...")
    
    try:
        from src.ai.multi_model import MultiModelManager, get_multi_model_manager
        
        multi_model = get_multi_model_manager("DEMO")
        print("[OK] Multi-model manager initialized")
        
        # Test getting adaptive weights
        adaptive_weights = multi_model.get_adaptive_weights()
        print(f"[OK] Retrieved adaptive weights: {adaptive_weights}")
        
        # Test updating weights from performance
        multi_model.update_weights_from_performance()
        print("[OK] Updated weights from performance")
        
        # Test logging model prediction
        multi_model.log_model_prediction(
            model_name="qwen2.5",
            symbol="TEST",
            predicted_probability=0.8,
            actual_outcome="PENDING"
        )
        print("[OK] Logged model prediction")
        
        # Test getting performance summary
        performance_summary = multi_model.get_performance_summary()
        print(f"[OK] Retrieved performance summary: {performance_summary}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Multi-model integration test failed: {e}")
        return False

def test_database_integration():
    """Test database integration for model performance."""
    print("\nTesting Database Integration...")
    
    try:
        from src.config.database import (
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

def test_weight_calculation_logic():
    """Test weight calculation and tie-break logic."""
    print("\nTesting Weight Calculation Logic...")
    
    try:
        from src.ai.adaptive_weights import AdaptiveWeightManager
        
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
        from src.ai.adaptive_weights import AdaptiveWeightManager, add_model_prediction
        
        # Create manager and add some data
        manager = AdaptiveWeightManager()
        
        base_date = datetime.now() - timedelta(days=3)
        add_model_prediction(
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
        assert len(new_manager.model_predictions) == 1, "Should have 1 prediction"
        assert new_manager.model_predictions[0]["model_name"] == "test_model", \
            "Should have correct model name"
        print("[OK] Imported data is correct")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Export/import functionality test failed: {e}")
        return False

def main():
    """Run all Phase 5 smoke tests."""
    print("=" * 60)
    print("PHASE 5 SMOKE TEST - ADAPTIVE ENSEMBLE WEIGHTS")
    print("=" * 60)
    
    start_time = time.time()
    
    tests = [
        ("Adaptive Weight Manager", test_adaptive_weight_manager),
        ("Enhanced Ensemble Integration", test_enhanced_ensemble_integration),
        ("Multi-Model Integration", test_multi_model_integration),
        ("Database Integration", test_database_integration),
        ("Weight Calculation Logic", test_weight_calculation_logic),
        ("Export/Import Functionality", test_export_import_functionality)
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
    print("PHASE 5 SMOKE TEST SUMMARY")
    print("=" * 60)
    print(f"Duration: {duration:.1f}s")
    print(f"Tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if failed == 0:
        print("\n[OK] ALL TESTS PASSED - Phase 5 implementation working correctly!")
        print("[READY] Ready for Phase 6: Drawdown-Aware Kelly & ATR Brackets!")
    else:
        print(f"\n[FAIL] {failed} TESTS FAILED - Check implementation")
    
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
