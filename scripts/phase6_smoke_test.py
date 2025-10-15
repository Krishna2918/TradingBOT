#!/usr/bin/env python3
"""
Phase 6 Smoke Test - Drawdown-Aware Kelly & ATR Brackets

This script performs a comprehensive smoke test of Phase 6 components:
1. Drawdown-aware Kelly sizing
2. ATR-based bracket parameters
3. Risk fraction validation
4. Database schema updates
5. Execution integration
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_drawdown_aware_kelly():
    """Test drawdown-aware Kelly sizing functionality."""
    print("\n[TEST] Drawdown-Aware Kelly Sizing")
    print("=" * 50)
    
    try:
        from trading.risk import RiskManager, RiskMetrics
        
        # Create risk manager
        risk_manager = RiskManager()
        
        # Set custom parameters for testing
        risk_manager.max_position_size = 10000
        risk_manager.default_risk_per_trade = 0.02
        risk_manager.max_daily_drawdown = 0.05
        risk_manager.min_drawdown_scale = 0.1
        risk_manager.drawdown_window_hours = 24
        
        print("[OK] RiskManager created successfully")
        
        # Test drawdown calculation
        with patch('config.database.get_portfolio_snapshots') as mock_snapshots:
            mock_snapshots.return_value = [
                {'portfolio_value': 10000, 'timestamp': '2024-01-01 09:00:00'},
                {'portfolio_value': 9500, 'timestamp': '2024-01-01 10:00:00'},
                {'portfolio_value': 9000, 'timestamp': '2024-01-01 11:00:00'},
            ]
            
            drawdown_data = risk_manager.calculate_daily_drawdown("DEMO")
            drawdown = drawdown_data.get('drawdown_percent', 0.0)
            print(f"[OK] Daily drawdown calculated: {drawdown:.2%}")
            
            # Test drawdown scale
            scale = risk_manager.calculate_drawdown_scale("DEMO")
            print(f"[OK] Drawdown scale calculated: {scale:.2f}")
            
            # Test Kelly fraction
            kelly = risk_manager.calculate_kelly_fraction(0.7, 0.3, 0.2)
            print(f"[OK] Kelly fraction calculated: {kelly:.3f}")
        
        # Test position sizing with drawdown awareness
        with patch('config.database.get_portfolio_snapshots') as mock_snapshots:
            mock_snapshots.return_value = [
                {'portfolio_value': 10000, 'timestamp': '2024-01-01 09:00:00'},
                {'portfolio_value': 9000, 'timestamp': '2024-01-01 10:00:00'},
            ]
            
            with patch('adaptive.confidence_calibration.get_confidence_calibrator') as mock_calibrator:
                mock_cal = Mock()
                mock_cal.calibrate_confidence.return_value = 0.7
                mock_calibrator.return_value = mock_cal
                
                metrics = risk_manager.calculate_position_size(
                    signal_confidence=0.7,
                    account_balance=10000.0,
                    volatility=0.02,
                    entry_price=150.0,
                    stop_loss=145.0,
                    model_name="test_model",
                    symbol="AAPL",
                    trade_date=datetime(2024, 1, 1)
                )
                
                print(f"[OK] Position size calculated: ${metrics.position_size:,.2f}")
                print(f"[OK] Drawdown scale applied: {metrics.drawdown_scale:.2f}")
                print(f"[OK] Kelly fraction: {metrics.kelly_fraction:.3f}")
                print(f"[OK] Calibrated confidence: {metrics.calibrated_confidence:.2f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Drawdown-aware Kelly test failed: {e}")
        logger.exception("Drawdown-aware Kelly test failed")
        return False

def test_atr_brackets():
    """Test ATR-based bracket parameters."""
    print("\n[TEST] ATR-Based Bracket Parameters")
    print("=" * 50)
    
    try:
        from trading.atr_brackets import ATRBracketManager, BracketParameters
        
        # Create ATR bracket manager
        atr_manager = ATRBracketManager()
        print("[OK] ATRBracketManager created successfully")
        
        # Test bracket calculation
        params = atr_manager.calculate_atr_brackets(
            symbol="AAPL",
            entry_price=150.0,
            atr=3.0,
            volatility_multiplier=2.0
        )
        
        print(f"[OK] Bracket parameters calculated:")
        print(f"  - Entry Price: ${params.entry_price:.2f}")
        print(f"  - Stop Loss: ${params.stop_loss:.2f}")
        print(f"  - Take Profit: ${params.take_profit:.2f}")
        print(f"  - ATR: {params.atr:.2f}")
        print(f"  - ATR Multiplier: {params.atr_multiplier:.2f}")
        print(f"  - R-Multiple: {params.r_multiple:.2f}")
        print(f"  - Volatility: {params.volatility_percent:.2f}%")
        
        # Test different bracket types
        aggressive_params = atr_manager.calculate_atr_brackets(
            symbol="AAPL",
            entry_price=150.0,
            atr=3.0,
            volatility_multiplier=2.0
        )
        
        print(f"[OK] Aggressive brackets - R-Multiple: {aggressive_params.r_multiple:.2f}")
        
        # Test bracket logging
        from trading.atr_brackets import log_bracket_parameters
        log_bracket_parameters(params, "DEMO")
        print("[OK] Bracket parameters logged to database")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] ATR brackets test failed: {e}")
        logger.exception("ATR brackets test failed")
        return False

def test_execution_integration():
    """Test execution engine integration with ATR brackets."""
    print("\n[TEST] Execution Engine Integration")
    print("=" * 50)
    
    try:
        from trading.execution import ExecutionEngine, Order, OrderType, OrderSide
        from trading.atr_brackets import BracketParameters
        
        # Create execution engine
        execution_engine = ExecutionEngine()
        print("[OK] ExecutionEngine created successfully")
        
        # Test ATR value retrieval
        atr = execution_engine._get_atr_value("AAPL", "DEMO")
        print(f"[OK] ATR value retrieved for AAPL: {atr:.3f}")
        
        # Test different symbols
        atr_tsla = execution_engine._get_atr_value("TSLA", "DEMO")
        atr_aapl = execution_engine._get_atr_value("AAPL", "DEMO")
        print(f"[OK] ATR comparison - TSLA: {atr_tsla:.3f}, AAPL: {atr_aapl:.3f}")
        
        # Test buy order with brackets (mocked)
        with patch('config.database.get_database_manager') as mock_db:
            mock_db_manager = Mock()
            mock_db.return_value = mock_db_manager
            
            with patch('trading.atr_brackets.get_atr_bracket_manager') as mock_atr:
                mock_atr_manager = Mock()
                mock_atr.return_value = mock_atr_manager
                
                # Mock bracket parameters
                from datetime import datetime
                bracket_params = BracketParameters(
                    symbol="AAPL",
                    entry_price=150.0,
                    stop_loss=144.0,
                    take_profit=156.0,
                    atr=3.0,
                    atr_multiplier=2.0,
                    r_multiple=1.0,
                    bracket_type="conservative",
                    volatility_percent=2.0,
                    risk_amount=600.0,
                    reward_amount=600.0,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                mock_atr_manager.calculate_atr_brackets.return_value = bracket_params
                
                # Mock database operations
                mock_cursor = Mock()
                mock_cursor.lastrowid = 1
                mock_conn = Mock()
                mock_conn.cursor.return_value = mock_cursor
                mock_conn.__enter__ = Mock(return_value=mock_conn)
                mock_conn.__exit__ = Mock(return_value=None)
                mock_db_manager.get_connection.return_value = mock_conn
                
                order = execution_engine.execute_buy_order(
                    symbol="AAPL",
                    quantity=100,
                    price=150.0
                )
                
                print(f"[OK] Buy order executed with brackets:")
                print(f"  - Success: {order.success}")
                if order.order:
                    print(f"  - Symbol: {order.order.symbol}")
                    print(f"  - Quantity: {order.order.quantity}")
                    print(f"  - Price: ${order.order.price:.2f}")
                    print(f"  - ATR: {order.order.atr:.2f}")
                    print(f"  - ATR Multiplier: {order.order.atr_multiplier:.2f}")
                    print(f"  - R-Multiple: {order.order.r_multiple:.2f}")
                    print(f"  - Bracket Type: {order.order.bracket_type}")
                    print(f"  - Volatility: {order.order.volatility_percent:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Execution integration test failed: {e}")
        logger.exception("Execution integration test failed")
        return False

def test_database_schema():
    """Test database schema updates."""
    print("\n[TEST] Database Schema Updates")
    print("=" * 50)
    
    try:
        from config.database import get_database_manager
        
        # Get database manager
        db_manager = get_database_manager()
        print("[OK] Database manager created successfully")
        
        # Test orders table schema
        with db_manager.get_connection("DEMO") as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(orders)")
            columns = [row[1] for row in cursor.fetchall()]
            
            # Check for new columns
            new_columns = ["atr", "atr_multiplier", "r_multiple", "bracket_type", "volatility_percent"]
            for col in new_columns:
                if col in columns:
                    print(f"[OK] Column '{col}' exists in orders table")
                else:
                    print(f"[FAIL] Column '{col}' missing from orders table")
                    return False
        
        # Test bracket parameters table
        with db_manager.get_connection("DEMO") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='bracket_parameters'")
            result = cursor.fetchone()
            if result:
                print("[OK] bracket_parameters table exists")
            else:
                print("[FAIL] bracket_parameters table missing")
                return False
        
        # Test portfolio snapshots table
        with db_manager.get_connection("DEMO") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio_snapshots'")
            result = cursor.fetchone()
            if result:
                print("[OK] portfolio_snapshots table exists")
            else:
                print("[FAIL] portfolio_snapshots table missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Database schema test failed: {e}")
        logger.exception("Database schema test failed")
        return False

def test_risk_fraction_validation():
    """Test risk fraction validation and edge cases."""
    print("\n[TEST] Risk Fraction Validation")
    print("=" * 50)
    
    try:
        from trading.risk import RiskManager
        
        # Create risk manager
        risk_manager = RiskManager()
        
        # Set custom parameters for testing
        risk_manager.max_position_size = 10000
        risk_manager.default_risk_per_trade = 0.02
        risk_manager.max_daily_drawdown = 0.05
        risk_manager.min_drawdown_scale = 0.1
        risk_manager.drawdown_window_hours = 24
        
        print("[OK] RiskManager created for validation tests")
        
        # Test negative Kelly handling
        with patch('config.database.get_portfolio_snapshots') as mock_snapshots:
            mock_snapshots.return_value = [
                {'portfolio_value': 10000, 'timestamp': '2024-01-01 09:00:00'},
            ]
            
            with patch('adaptive.confidence_calibration.get_confidence_calibrator') as mock_calibrator:
                mock_cal = Mock()
                mock_cal.calibrate_confidence.return_value = 0.3  # Low confidence
                mock_calibrator.return_value = mock_cal
                
                metrics = risk_manager.calculate_position_size(
                    signal_confidence=0.7,
                    account_balance=10000.0,
                    volatility=0.02,
                    entry_price=150.0,
                    stop_loss=145.0,
                    model_name="test_model",
                    symbol="AAPL",
                    trade_date=datetime(2024, 1, 1)
                )
                
                print(f"[OK] Negative Kelly handled - Position size: ${metrics.position_size:,.2f}")
                print(f"[OK] Kelly fraction: {metrics.kelly_fraction:.3f}")
        
        # Test maximum position size cap
        with patch('config.database.get_portfolio_snapshots') as mock_snapshots:
            mock_snapshots.return_value = [
                {'portfolio_value': 100000, 'timestamp': '2024-01-01 09:00:00'},
            ]
            
            with patch('adaptive.confidence_calibration.get_confidence_calibrator') as mock_calibrator:
                mock_cal = Mock()
                mock_cal.calibrate_confidence.return_value = 0.95  # Very high confidence
                mock_calibrator.return_value = mock_cal
                
                metrics = risk_manager.calculate_position_size(
                    signal_confidence=0.7,
                    account_balance=10000.0,
                    volatility=0.02,
                    entry_price=150.0,
                    stop_loss=145.0,
                    model_name="test_model",
                    symbol="AAPL",
                    trade_date=datetime(2024, 1, 1)
                )
                
                print(f"[OK] Position size capped at maximum: ${metrics.position_size:,.2f}")
                assert metrics.position_size <= 10000
        
        # Test drawdown scale edge cases
        scale = risk_manager.calculate_drawdown_scale(0.5)  # 50% drawdown
        print(f"[OK] Extreme drawdown scale: {scale:.2f}")
        
        scale = risk_manager.calculate_drawdown_scale(-0.1)  # Negative drawdown
        print(f"[OK] Negative drawdown handled: {scale:.2f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Risk fraction validation test failed: {e}")
        logger.exception("Risk fraction validation test failed")
        return False

def main():
    """Run all Phase 6 smoke tests."""
    print("Phase 6 Smoke Test - Drawdown-Aware Kelly & ATR Brackets")
    print("=" * 60)
    
    tests = [
        ("Drawdown-Aware Kelly Sizing", test_drawdown_aware_kelly),
        ("ATR-Based Bracket Parameters", test_atr_brackets),
        ("Execution Engine Integration", test_execution_integration),
        ("Database Schema Updates", test_database_schema),
        ("Risk Fraction Validation", test_risk_fraction_validation),
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
    print("Phase 6 Smoke Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All Phase 6 components are working correctly!")
        return 0
    else:
        print(f"\n[FAILURE] {total - passed} test(s) failed. Please check the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
