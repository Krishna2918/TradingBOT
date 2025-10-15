#!/usr/bin/env python3
"""
Phase 7 Smoke Test - Regime Awareness

This script performs a comprehensive smoke test of Phase 7 components:
1. Regime detection and classification
2. Regime-aware ensemble weights
3. Regime-aware Kelly adjustments
4. Regime-aware ATR brackets
5. Regime policy management
6. Database integration
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import MarketRegime at module level for use in tests
from ai.regime_detection import MarketRegime

def test_regime_detection():
    """Test regime detection functionality."""
    print("\n[TEST] Regime Detection")
    print("=" * 50)
    
    try:
        from ai.regime_detection import RegimeDetector, MarketRegime, TrendDirection
        
        # Create regime detector
        detector = RegimeDetector()
        print("[OK] RegimeDetector created successfully")
        
        # Test regime classification
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
        
        # Test transition probability
        prob = detector._calculate_transition_probability(metrics, regime)
        print(f"[OK] Transition probability: {prob:.3f}")
        
        # Test ATR calculation
        data = pd.DataFrame({
            'high': [100, 102, 101, 103, 105],
            'low': [98, 99, 100, 101, 102],
            'close': [99, 101, 100, 102, 104]
        })
        atr = detector._calculate_atr(data)
        print(f"[OK] ATR calculation: {atr:.3f}")
        
        # Test trend slope calculation
        prices = pd.Series([100, 101, 102, 103, 104, 105])
        slope = detector._calculate_trend_slope(prices)
        print(f"[OK] Trend slope: {slope:.6f}")
        
        # Test trend consistency
        consistency = detector._calculate_trend_consistency(prices)
        print(f"[OK] Trend consistency: {consistency:.3f}")
        
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
            MarketRegime.TRENDING_HIGH_VOL,
            MarketRegime.CHOPPY_LOW_VOL,
            MarketRegime.CHOPPY_HIGH_VOL,
            MarketRegime.TRANSITION
        ]
        
        for regime in regimes_to_test:
            weights = policy_manager.get_ensemble_weights(regime)
            total_weight = sum(weights.values())
            print(f"[OK] {regime.value} ensemble weights sum: {total_weight:.3f}")
            
            # Verify weights sum to approximately 1
            assert abs(total_weight - 1.0) < 0.01
        
        # Test Kelly adjustments
        kelly_adj = policy_manager.get_kelly_adjustments(MarketRegime.CHOPPY_HIGH_VOL)
        print(f"[OK] Choppy high vol Kelly multiplier: {kelly_adj['kelly_multiplier']:.2f}")
        print(f"[OK] Choppy high vol max risk: {kelly_adj['max_position_risk']:.3f}")
        print(f"[OK] Choppy high vol confidence threshold: {kelly_adj['confidence_threshold']:.2f}")
        
        # Test ATR brackets
        atr_brackets = policy_manager.get_atr_brackets(MarketRegime.TRENDING_LOW_VOL)
        print(f"[OK] Trending low vol ATR multiplier: {atr_brackets['atr_multiplier']:.2f}")
        print(f"[OK] Trending low vol R-multiple: {atr_brackets['r_multiple']:.2f}")
        
        # Test lookback windows
        lookbacks = policy_manager.get_lookback_windows(MarketRegime.TRANSITION)
        print(f"[OK] Transition price lookback: {lookbacks['price_lookback']} days")
        print(f"[OK] Transition sentiment lookback: {lookbacks['sentiment_lookback']} days")
        
        # Test position management
        position_mgmt = policy_manager.get_position_management(MarketRegime.TRENDING_LOW_VOL)
        print(f"[OK] Trending low vol max positions: {position_mgmt['max_positions']}")
        print(f"[OK] Trending low vol max holding days: {position_mgmt['max_holding_days']}")
        
        # Test risk management
        risk_mgmt = policy_manager.get_risk_management(MarketRegime.TRANSITION)
        print(f"[OK] Transition max portfolio risk: {risk_mgmt['max_portfolio_risk']:.3f}")
        print(f"[OK] Transition daily loss limit: {risk_mgmt['daily_loss_limit']:.3f}")
        
        # Test feature flags
        feature_flags = [
            'enable_regime_weights',
            'enable_regime_kelly',
            'enable_regime_brackets',
            'enable_regime_lookbacks',
            'enable_regime_risk',
            'enable_regime_positions',
            'enable_regime_transitions'
        ]
        
        for flag in feature_flags:
            enabled = policy_manager.is_feature_enabled(flag)
            print(f"[OK] Feature flag {flag}: {enabled}")
        
        # Test transition rules
        rules = policy_manager.get_transition_rules()
        print(f"[OK] Min regime duration: {rules['min_regime_duration']} days")
        print(f"[OK] Regime change threshold: {rules['regime_change_threshold']:.2f}")
        print(f"[OK] Transition smoothing: {rules['transition_smoothing']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Regime policy management test failed: {e}")
        logger.exception("Regime policy management test failed")
        return False

def test_regime_aware_ensemble():
    """Test regime-aware ensemble functionality."""
    print("\n[TEST] Regime-Aware Ensemble")
    print("=" * 50)
    
    try:
        # Mock the regime detection and policy functions before importing
        with patch('ai.regime_detection.detect_current_regime') as mock_regime:
            with patch('config.regime_policy_manager.get_ensemble_weights') as mock_regime_weights:
                with patch('ai.adaptive_weights.get_ensemble_weights') as mock_adaptive_weights:
                    # Mock regime detection
                    mock_regime.return_value = Mock(regime=MarketRegime.TRENDING_LOW_VOL)
                    
                    # Mock regime weights
                    mock_regime_weights.return_value = {
                        'technical_analyst': 0.35,
                        'sentiment_analyst': 0.20,
                        'fundamental_analyst': 0.25,
                        'risk_analyst': 0.10,
                        'market_regime_analyst': 0.10
                    }
                    
                    # Mock adaptive weights
                    mock_adaptive_weights.return_value = {
                        'technical_analyst': 0.30,
                        'sentiment_analyst': 0.25,
                        'fundamental_analyst': 0.20,
                        'risk_analyst': 0.15,
                        'market_regime_analyst': 0.10
                    }
                    
                    from ai.enhanced_ensemble import EnhancedEnsemble
                    
                    # Create ensemble
                    ensemble = EnhancedEnsemble()
                    print("[OK] EnhancedEnsemble created successfully")
                    
                    # Mock market analysis
                    market_analysis = Mock()
                    market_analysis.technical_indicators = {'overall_score': 0.7}
                    market_analysis.sentiment_score = 0.6
                    market_analysis.fundamental_score = 0.8
                    market_analysis.market_regime = "TRENDING"
                    market_analysis.volatility = 0.02
                    
                    # Test ensemble analysis
                    result = ensemble._run_ensemble_analysis(market_analysis, "ENTRY")
                    
                    print(f"[OK] Ensemble analysis result: {result.get('action', 'N/A')}")
                    print(f"[OK] Ensemble confidence: {result.get('confidence', 0):.3f}")
                    
                    # Verify regime weights were called
                    mock_regime_weights.assert_called_once_with(MarketRegime.TRENDING_LOW_VOL)
                    print("[OK] Regime weights integration verified")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Regime-aware ensemble test failed: {e}")
        logger.exception("Regime-aware ensemble test failed")
        return False

def test_regime_aware_risk_management():
    """Test regime-aware risk management."""
    print("\n[TEST] Regime-Aware Risk Management")
    print("=" * 50)
    
    try:
        # Mock the regime detection and policy functions before importing
        with patch('ai.regime_detection.detect_current_regime') as mock_regime:
            with patch('config.regime_policy_manager.get_kelly_adjustments') as mock_adjustments:
                with patch('adaptive.confidence_calibration.get_confidence_calibrator') as mock_calibrator:
                    # Mock regime detection
                    mock_regime.return_value = Mock(regime=MarketRegime.CHOPPY_HIGH_VOL)
                    
                    # Mock Kelly adjustments
                    mock_adjustments.return_value = {
                        'kelly_multiplier': 0.6,
                        'max_position_risk': 0.01,
                        'confidence_threshold': 0.8
                    }
                    
                    # Mock confidence calibrator
                    mock_cal = Mock()
                    mock_cal.calibrate_confidence.return_value = 0.75
                    mock_calibrator.return_value = mock_cal
                    
                    from trading.risk import RiskManager
                    
                    # Create risk manager
                    risk_manager = RiskManager()
                    print("[OK] RiskManager created successfully")
                    
                    # Mock database calls
                    with patch('config.database.execute_query') as mock_query:
                        mock_query.return_value = []
                        
                        # Test position sizing
                        metrics = risk_manager.calculate_position_size(
                            signal_confidence=0.75,
                            account_balance=10000.0,
                            volatility=0.02,
                            entry_price=100.0,
                            stop_loss=98.0,
                            model_name="test_model",
                            symbol="AAPL",
                            trade_date=datetime.now()
                        )
                        
                        print(f"[OK] Position size: ${metrics.position_size:,.2f}")
                        print(f"[OK] Risk amount: ${metrics.risk_amount:,.2f}")
                        print(f"[OK] Calibrated confidence: {metrics.calibrated_confidence:.3f}")
                        print(f"[OK] Kelly fraction: {metrics.kelly_fraction:.3f}")
                        
                        # Verify regime adjustments were called
                        mock_adjustments.assert_called_once_with(MarketRegime.CHOPPY_HIGH_VOL)
                        print("[OK] Regime Kelly adjustments integration verified")
        
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
        # Mock the regime detection and policy functions before importing
        with patch('ai.regime_detection.detect_current_regime') as mock_regime:
            with patch('config.regime_policy_manager.get_atr_brackets') as mock_brackets:
                # Mock regime detection
                mock_regime.return_value = Mock(regime=MarketRegime.TRENDING_HIGH_VOL)
                
                # Mock bracket parameters
                mock_brackets.return_value = {
                    'atr_multiplier': 2.5,
                    'r_multiple': 1.5,
                    'min_stop_loss_percent': 0.02,
                    'max_stop_loss_percent': 0.12
                }
                
                from trading.atr_brackets import ATRBracketManager
                
                # Create ATR bracket manager
                atr_manager = ATRBracketManager()
                print("[OK] ATRBracketManager created successfully")
                
                # Test bracket calculation
                params = atr_manager.calculate_atr_brackets(
                    symbol="AAPL",
                    entry_price=150.0,
                    atr=3.0
                )
                
                print(f"[OK] Entry price: ${params.entry_price:.2f}")
                print(f"[OK] Stop loss: ${params.stop_loss:.2f}")
                print(f"[OK] Take profit: ${params.take_profit:.2f}")
                print(f"[OK] ATR multiplier: {params.atr_multiplier:.2f}")
                print(f"[OK] R-multiple: {params.r_multiple:.2f}")
                print(f"[OK] Volatility: {params.volatility_percent:.2f}%")
                
                # Verify regime brackets were called
                mock_brackets.assert_called_once_with(MarketRegime.TRENDING_HIGH_VOL)
                print("[OK] Regime ATR brackets integration verified")
        
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
            
            # Check for required columns
            required_columns = [
                'id', 'timestamp', 'symbol', 'regime', 'trend_direction',
                'volatility_level', 'trend_strength', 'volatility_ratio',
                'atr_percentile', 'regime_confidence', 'transition_probability',
                'mode', 'created_at'
            ]
            
            for column in required_columns:
                if column in columns:
                    print(f"[OK] Column '{column}' exists in regime_state table")
                else:
                    print(f"[FAIL] Column '{column}' missing from regime_state table")
                    return False
        
        # Test regime_state indexes
        with db_manager.get_connection("DEMO") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='regime_state'")
            indexes = [row[0] for row in cursor.fetchall()]
            
            # Check for required indexes
            required_indexes = [
                'idx_regime_state_symbol',
                'idx_regime_state_timestamp',
                'idx_regime_state_regime',
                'idx_regime_state_mode'
            ]
            
            for index in required_indexes:
                if index in indexes:
                    print(f"[OK] Index '{index}' exists")
                else:
                    print(f"[FAIL] Index '{index}' missing")
                    return False
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Database integration test failed: {e}")
        logger.exception("Database integration test failed")
        return False

def test_regime_transition_smoothing():
    """Test regime transition smoothing functionality."""
    print("\n[TEST] Regime Transition Smoothing")
    print("=" * 50)
    
    try:
        from ai.regime_detection import RegimeDetector, MarketRegime
        
        # Create regime detector
        detector = RegimeDetector()
        print("[OK] RegimeDetector created for transition testing")
        
        # Test transition probability calculation
        metrics = Mock()
        metrics.atr_percentile = 0.7  # Near volatility threshold
        metrics.trend_consistency = 0.3  # Near trend threshold
        
        # Test transition regime
        prob = detector._calculate_transition_probability(metrics, MarketRegime.TRANSITION)
        print(f"[OK] Transition regime probability: {prob:.3f}")
        
        # Test non-transition regime
        prob = detector._calculate_transition_probability(metrics, MarketRegime.TRENDING_LOW_VOL)
        print(f"[OK] Trending low vol transition probability: {prob:.3f}")
        
        # Test confidence variation
        metrics.atr_percentile = 0.9  # Clear high volatility
        metrics.trend_consistency = 0.8  # Clear strong trend
        confidence = detector._calculate_regime_confidence(metrics, MarketRegime.TRENDING_HIGH_VOL)
        print(f"[OK] High confidence scenario: {confidence:.3f}")
        
        # Test low confidence scenario
        metrics.atr_percentile = 0.5  # Neutral volatility
        metrics.trend_consistency = 0.3  # Weak trend
        confidence = detector._calculate_regime_confidence(metrics, MarketRegime.TRANSITION)
        print(f"[OK] Low confidence scenario: {confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Regime transition smoothing test failed: {e}")
        logger.exception("Regime transition smoothing test failed")
        return False

def main():
    """Run all Phase 7 smoke tests."""
    print("Phase 7 Smoke Test - Regime Awareness")
    print("=" * 60)
    
    tests = [
        ("Regime Detection", test_regime_detection),
        ("Regime Policy Management", test_regime_policy_management),
        ("Regime-Aware Ensemble", test_regime_aware_ensemble),
        ("Regime-Aware Risk Management", test_regime_aware_risk_management),
        ("Regime-Aware ATR Brackets", test_regime_aware_atr_brackets),
        ("Database Integration", test_database_integration),
        ("Regime Transition Smoothing", test_regime_transition_smoothing),
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
    print("Phase 7 Smoke Test Summary")
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
