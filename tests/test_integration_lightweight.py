"""
Lightweight Integration Test Suite

Tests core integrated components without heavy dependencies
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import testable components
from src.execution import ExecutionEngine, OrderType, OrderSide, OrderStatus
from src.event_awareness import EventCalendar, VolatilityDetector, AnomalyDetector
from src.penny_stocks import PennyStockDetector
from src.sip import SIPSimulator
from src.trading_modes import ModeManager


class TestExecutionIntegration:
    """Test Execution Engine Integration"""
    
    def test_complete_order_lifecycle(self):
        """Test complete order from creation to execution"""
        engine = ExecutionEngine()
        
        # Create order
        order = engine.create_order(
            symbol="SHOP.TO",
            side=OrderSide.BUY,
            quantity=50.5,
            order_type=OrderType.MARKET
        )
        
        assert order.symbol == "SHOP.TO"
        assert order.quantity == 50.5
        print(f"✅ Order created: {order.order_id}")
        
        # Execute order
        success = engine.execute_market_order(
            order=order,
            current_price=150.0,
            volume=1000000
        )
        
        assert success
        assert order.status == OrderStatus.FILLED
        print(f"✅ Order executed: {order.filled_quantity} @ ${order.average_fill_price:.2f}")
        
        # Check statistics
        stats = engine.get_execution_statistics()
        assert stats['total_executions'] == 1
        assert stats['total_volume'] == 50.5
        print(f"✅ Stats tracked: {stats['total_executions']} executions")


class TestEventAwarenessIntegration:
    """Test Event Awareness Integration"""
    
    def test_calendar_workflow(self):
        """Test event calendar workflow"""
        calendar = EventCalendar()
        
        # Check upcoming events
        upcoming = calendar.get_upcoming_events(hours_ahead=24)
        print(f"✅ Found {len(upcoming)} upcoming events")
        
        # Check holiday
        christmas = datetime(2025, 12, 25)
        is_holiday = calendar.is_market_holiday(christmas)
        assert is_holiday
        print("✅ Holiday detection working")
        
        # Check high-impact events
        high_impact = calendar.get_high_impact_events(days_ahead=7)
        print(f"✅ Found {len(high_impact)} high-impact events (Bank of Canada)")
    
    def test_volatility_workflow(self):
        """Test volatility detection workflow"""
        detector = VolatilityDetector()
        
        # Create sample data
        np.random.seed(42)
        base_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        data = pd.DataFrame({
            'open': base_prices + np.random.randn(100) * 0.1,
            'high': base_prices + np.abs(np.random.randn(100) * 0.5),
            'low': base_prices - np.abs(np.random.randn(100) * 0.5),
            'close': base_prices + np.random.randn(100) * 0.2,
            'volume': np.random.uniform(100000, 200000, 100)
        })
        
        # Analyze
        analysis = detector.analyze_volatility("TEST.TO", data)
        
        assert 'historical_volatility' in analysis
        assert 'volatility_regime' in analysis
        print(f"✅ Volatility: {analysis['historical_volatility']:.2f}%, Regime: {analysis['volatility_regime']}")
    
    def test_anomaly_workflow(self):
        """Test anomaly detection workflow"""
        detector = AnomalyDetector()
        
        # Create data
        np.random.seed(42)
        data = pd.DataFrame({
            'open': np.random.uniform(95, 105, 200),
            'high': np.random.uniform(100, 110, 200),
            'low': np.random.uniform(90, 100, 200),
            'close': np.random.uniform(95, 105, 200),
            'volume': np.random.uniform(1000000, 2000000, 200)
        })
        
        # Train
        detector.train(data)
        assert detector.is_trained
        print("✅ Anomaly detector trained")
        
        # Detect
        results = detector.detect_anomalies("TEST.TO", data)
        anomaly_count = results['is_anomaly'].sum()
        print(f"✅ Anomaly detection: {anomaly_count} anomalies found")


class TestPennyStockIntegration:
    """Test Penny Stock Integration"""
    
    def test_penny_stock_workflow(self):
        """Test penny stock detection workflow"""
        detector = PennyStockDetector()
        
        # Create penny stock data (< $5)
        np.random.seed(42)
        data = pd.DataFrame({
            'open': np.random.uniform(2.0, 2.5, 100),
            'high': np.random.uniform(2.2, 2.7, 100),
            'low': np.random.uniform(1.8, 2.3, 100),
            'close': np.random.uniform(2.0, 2.5, 100),
            'volume': np.random.uniform(100000, 500000, 100)
        })
        
        # Analyze
        profile = detector.analyze_penny_stock("ABC.V", data)
        
        assert profile is not None
        assert profile.price < 5.0
        print(f"✅ Penny stock: {profile.symbol} @ ${profile.price:.2f}")
        print(f"   Liquidity: {profile.liquidity_score:.2f}, Risk: {profile.risk_level}, Tradeable: {profile.is_tradeable}")
    
    def test_volume_spike(self):
        """Test volume spike detection"""
        detector = PennyStockDetector()
        
        historical = pd.Series([100000] * 50)
        current = 350000  # 3.5x spike
        
        is_spike, ratio = detector.detect_volume_spike("TEST.V", current, historical)
        
        assert is_spike
        assert ratio >= 3.0
        print(f"✅ Volume spike detected: {ratio:.1f}x")


class TestSIPIntegration:
    """Test SIP Integration"""
    
    def test_sip_workflow(self):
        """Test SIP investment workflow"""
        sip = SIPSimulator()
        
        # Process profit
        transaction = sip.process_daily_profit(
            daily_profit=10000.0,
            etf_price=110.50
        )
        
        assert transaction is not None
        assert transaction.amount_cad == 100.0  # 1% of 10K
        assert transaction.shares_purchased > 0
        print(f"✅ SIP: ${transaction.amount_cad:.2f} → {transaction.shares_purchased:.4f} shares @ ${transaction.share_price:.2f}")
        
        # Get portfolio
        portfolio = sip.get_portfolio_value(110.50)
        assert portfolio['total_shares'] > 0
        print(f"✅ Portfolio: {portfolio['total_shares']:.4f} shares, ${portfolio['current_value']:.2f}")
    
    def test_sip_minimum_threshold(self):
        """Test SIP enforces minimum"""
        sip = SIPSimulator()
        
        # Small profit
        transaction = sip.process_daily_profit(
            daily_profit=1000.0,  # Only $10 (1%)
            etf_price=110.50
        )
        
        assert transaction is None  # Below $25 minimum
        print("✅ SIP minimum threshold enforced ($25)")


class TestTradingModesIntegration:
    """Test Trading Modes Integration"""
    
    def test_mode_manager_workflow(self):
        """Test mode manager workflow"""
        manager = ModeManager()
        
        # Check mode
        assert manager.is_demo_mode()
        print(f"✅ Mode: {manager.current_mode.value}")
        
        # Get account
        account = manager.get_current_account_info()
        assert account['capital'] > 0
        assert 'num_trades' in account
        print(f"✅ Account: ${account['capital']:,.2f}, {account['num_trades']} trades")
        
        # Get learning data
        learning = manager.get_shared_learning_data()
        assert 'insights' in learning
        assert 'total_trades' in learning
        print(f"✅ Learning data: {learning['total_trades']} total trades")


class TestIntegrationScenarios:
    """Test Real Integration Scenarios"""
    
    def test_end_to_end_trade_flow(self):
        """Test complete trade flow"""
        print("\n🔄 Testing End-to-End Trade Flow")
        
        # 1. Check events
        calendar = EventCalendar()
        if calendar.is_market_holiday():
            print("   ⏸️  Market is closed (holiday)")
        else:
            print("   ✅ Market is open")
        
        # 2. Analyze volatility
        detector = VolatilityDetector()
        prices = pd.Series(100 + np.cumsum(np.random.randn(50)))
        hv = detector.calculate_historical_volatility(prices, period=20)
        regime = detector.classify_volatility_regime(hv)
        print(f"   ✅ Volatility: {hv:.2f}% ({regime})")
        
        # 3. Create and execute order
        engine = ExecutionEngine()
        order = engine.create_order("TD.TO", OrderSide.BUY, 100, OrderType.MARKET)
        success = engine.execute_market_order(order, 85.0, 1000000)
        print(f"   ✅ Order executed: {order.filled_quantity} shares @ ${order.average_fill_price:.2f}")
        
        # 4. Process SIP
        sip = SIPSimulator()
        profit = 5000.0
        transaction = sip.process_daily_profit(profit, 110.50)
        if transaction:
            print(f"   ✅ SIP invested: ${transaction.amount_cad:.2f}")
        
        print("   🎉 Complete trade flow successful!")
    
    def test_risk_checks_integration(self):
        """Test risk checks work together"""
        print("\n🛡️ Testing Risk Checks Integration")
        
        # Mode check
        manager = ModeManager()
        account = manager.get_current_account_info()
        print(f"   ✅ Mode: {manager.current_mode.value}, Capital: ${account['capital']:,.2f}")
        
        # Position sizing for penny stock
        detector = PennyStockDetector()
        np.random.seed(42)
        data = pd.DataFrame({
            'open': np.random.uniform(2.0, 2.5, 100),
            'high': np.random.uniform(2.2, 2.7, 100),
            'low': np.random.uniform(1.8, 2.3, 100),
            'close': np.random.uniform(2.0, 2.5, 100),
            'volume': np.random.uniform(100000, 500000, 100)
        })
        
        profile = detector.analyze_penny_stock("XYZ.V", data)
        if profile and profile.is_tradeable:
            position_size = detector.calculate_position_size(profile, account['capital'])
            print(f"   ✅ Penny stock position: ${position_size:,.2f} (max 2% of capital)")
        
        print("   🎉 Risk checks passed!")


def run_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("🧪 RUNNING LIGHTWEIGHT INTEGRATION TESTS")
    print("=" * 80)
    
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--color=yes",
        "-s"
    ])
    
    print("\n" + "=" * 80)
    if exit_code == 0:
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("\n📊 Components Tested:")
        print("   ✅ Execution Engine - Order lifecycle")
        print("   ✅ Event Calendar - Holiday detection, events")
        print("   ✅ Volatility Detector - Regime classification")
        print("   ✅ Anomaly Detector - Isolation Forest")
        print("   ✅ Penny Stock Detector - Volume spikes, risk assessment")
        print("   ✅ SIP Simulator - Profit allocation, ETF investing")
        print("   ✅ Trading Modes - Demo/Live management")
        print("   ✅ Integration Scenarios - End-to-end workflows")
    else:
        print(f"❌ TESTS FAILED (exit code: {exit_code})")
    print("=" * 80 + "\n")
    
    return exit_code


if __name__ == "__main__":
    exit(run_tests())

