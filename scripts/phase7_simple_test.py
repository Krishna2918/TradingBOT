#!/usr/bin/env python3
"""
Phase 7 Simple Test - Regime Awareness

This script performs a simplified test of Phase 7 components focusing on core functionality.
"""

import sys
import os
import logging
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_regime_detection():
    """Test regime detection functionality."""
    print("\n[TEST] Regime Detection")
    print("=" * 50)
    
    try:
        from ai.regime_detection import RegimeDetector, MarketRegime
        
        # Create regime detector
        detector = RegimeDetector()
        print("[OK] RegimeDetector created successfully")
        
        # Test regime classification
        from unittest.mock import Mock
        metrics = Mock()
        metrics.atr_percentile = 0.3  # Low volatility
        metrics.trend_consistency = 0.8  # Strong trend
        
        regime = detector._classify_regime(metrics)
        print(f"[OK] Regime classification: {regime.value}")
        
        # Test trend direction detection
        metrics.price_trend_slope = 0.005
        direction = detector._determine_trend_direction(metrics)
        print(f"[OK] Trend direction: {direction.value}")
        
        # Test confidence calculation
        confidence = detector._calculate_regime_confidence(metrics, regime)
        print(f"[OK] Regime confidence: {confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Regime detection test failed: {e}")
        logger.exception("Regime detection test failed")
        return False

def test_regime_policy_management():
    """Test regime policy management."""
    print("\n[TEST] Regime Policy Management")
    print("=" * 50)
    
    try:
        from config.regime_policy_manager import RegimePolicyManager
        from ai.regime_detection import MarketRegime
        
        # Create policy manager
        policy_manager = RegimePolicyManager()
        print("[OK] RegimePolicyManager created successfully")
        
        # Test ensemble weights for different regimes
        regimes_to_test = [
            MarketRegime.TRENDING_LOW_VOL,
            MarketRegime.CHOPPY_HIGH_VOL,
            MarketRegime.TRANSITION
        ]
        
        for regime in regimes_to_test:
            weights = policy_manager.get_ensemble_weights(regime)
            total_weight = sum(weights.values())
            print(f"[OK] {regime.value} ensemble weights sum: {total_weight:.3f}")
        
        # Test Kelly adjustments
        kelly_adj = policy_manager.get_kelly_adjustments(MarketRegime.CHOPPY_HIGH_VOL)
        print(f"[OK] Choppy high vol Kelly multiplier: {kelly_adj['kelly_multiplier']:.2f}")
        print(f"[OK] Choppy high vol max risk: {kelly_adj['max_position_risk']:.3f}")
        
        # Test ATR brackets
        atr_brackets = policy_manager.get_atr_brackets(MarketRegime.TRENDING_LOW_VOL)
        print(f"[OK] Trending low vol ATR multiplier: {atr_brackets['atr_multiplier']:.2f}")
        print(f"[OK] Trending low vol R-multiple: {atr_brackets['r_multiple']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Regime policy management test failed: {e}")
        logger.exception("Regime policy management test failed")
        return False

def test_regime_aware_risk_management():
    """Test regime-aware risk management."""
    print("\n[TEST] Regime-Aware Risk Management")
    print("=" * 50)
    
    try:
        from trading.risk import RiskManager
        
        # Create risk manager
        risk_manager = RiskManager()
        print("[OK] RiskManager created successfully")
        
        # Test position sizing with different confidence levels
        test_cases = [
            (0.75, "High confidence"),
            (0.65, "Medium confidence"),
            (0.55, "Low confidence")
        ]
        
        for confidence, description in test_cases:
            metrics = risk_manager.calculate_position_size(
                signal_confidence=confidence,
                account_balance=10000.0,
                volatility=0.02,
                entry_price=100.0,
                stop_loss=98.0,
                model_name="test_model",
                symbol="AAPL",
                trade_date=datetime.now()
            )
            
            print(f"[OK] {description}: Position size ${metrics.position_size:,.2f}, "
                  f"Risk ${metrics.risk_amount:,.2f}, Kelly {metrics.kelly_fraction:.3f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Regime-aware risk management test failed: {e}")
        logger.exception("Regime-aware risk management test failed")
        return False

def test_regime_aware_atr_brackets():
    """Test regime-aware ATR brackets."""
    print("\n[TEST] Regime-Aware ATR Brackets")
    print("=" * 50)
    
    try:
        from trading.atr_brackets import ATRBracketManager
        
        # Create ATR bracket manager
        atr_manager = ATRBracketManager()
        print("[OK] ATRBracketManager created successfully")
        
        # Test bracket calculation with different ATR values
        test_cases = [
            (2.0, "Low volatility"),
            (3.0, "Medium volatility"),
            (5.0, "High volatility")
        ]
        
        for atr, description in test_cases:
            params = atr_manager.calculate_atr_brackets(
                symbol="AAPL",
                entry_price=150.0,
                atr=atr
            )
            
            print(f"[OK] {description}: SL ${params.stop_loss:.2f}, "
                  f"TP ${params.take_profit:.2f}, R {params.r_multiple:.2f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Regime-aware ATR brackets test failed: {e}")
        logger.exception("Regime-aware ATR brackets test failed")
        return False

def test_database_integration():
    """Test database integration for regime state."""
    print("\n[TEST] Database Integration")
    print("=" * 50)
    
    try:
        from config.database import get_database_manager
        
        # Get database manager
        db_manager = get_database_manager()
        print("[OK] Database manager created successfully")
        
        # Test regime_state table exists
        with db_manager.get_connection("DEMO") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='regime_state'")
            result = cursor.fetchone()
            if result:
                print("[OK] regime_state table exists")
            else:
                print("[FAIL] regime_state table missing")
                return False
        
        # Test regime_state table schema
        with db_manager.get_connection("DEMO") as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(regime_state)")
            columns = [row[1] for row in cursor.fetchall()]
            
            # Check for key columns
            key_columns = ['regime', 'trend_direction', 'volatility_level', 'regime_confidence']
            for column in key_columns:
                if column in columns:
                    print(f"[OK] Column '{column}' exists")
                else:
                    print(f"[FAIL] Column '{column}' missing")
                    return False
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Database integration test failed: {e}")
        logger.exception("Database integration test failed")
        return False

def main():
    """Run all Phase 7 simple tests."""
    print("Phase 7 Simple Test - Regime Awareness")
    print("=" * 60)
    
    tests = [
        ("Regime Detection", test_regime_detection),
        ("Regime Policy Management", test_regime_policy_management),
        ("Regime-Aware Risk Management", test_regime_aware_risk_management),
        ("Regime-Aware ATR Brackets", test_regime_aware_atr_brackets),
        ("Database Integration", test_database_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n[READY] Starting {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"[PASS] {test_name} test completed successfully")
            else:
                print(f"[FAIL] {test_name} test failed")
        except Exception as e:
            print(f"[ERROR] {test_name} test encountered an error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Phase 7 Simple Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All Phase 7 components are working correctly!")
        return 0
    else:
        print(f"\n[FAILURE] {total - passed} test(s) failed. Please check the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
