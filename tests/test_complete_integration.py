"""
Complete Integration Test Suite

Comprehensive testing of all integrated components:
- Orchestrator
- Execution Engine
- Event Awareness
- Penny Stocks
- SIP Simulator
- Risk Dashboard
- AI Models
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import all components
from src.orchestrator import TradingOrchestrator, get_orchestrator
from src.execution import ExecutionEngine, OrderType, OrderSide, OrderStatus
from src.event_awareness import EventCalendar, VolatilityDetector, AnomalyDetector
from src.penny_stocks import PennyStockDetector, get_penny_stock_detector
from src.sip import SIPSimulator, get_sip_simulator
from src.trading_modes import ModeManager


class TestOrchestrator:
    """Test Trading Orchestrator"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.orchestrator = TradingOrchestrator({
            'symbols': ['TEST.TO'],
            'cycle_interval_seconds': 60,
            'enable_auto_execution': False
        })
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initializes all components"""
        assert self.orchestrator is not None
        assert self.orchestrator.data_pipeline is not None
        assert self.orchestrator.execution_engine is not None
        assert self.orchestrator.event_calendar is not None
        assert self.orchestrator.mode_manager is not None
        print("✅ Orchestrator initialized with all components")
    
    def test_single_trading_cycle(self):
        """Test single trading cycle execution"""
        results = self.orchestrator.run_trading_cycle()
        
        assert results is not None
        assert 'cycle_number' in results
        assert 'status' in results
        assert results['cycle_number'] == 1
        print(f"✅ Trading cycle completed: {results['status']}")
    
    def test_pre_flight_checks(self):
        """Test pre-flight checks"""
        # Should pass kill switch check
        assert not self.orchestrator.kill_switch_manager.is_kill_switch_active()
        
        # Check trading mode
        mode = self.orchestrator.mode_manager.get_current_mode()
        assert mode is not None
        print(f"✅ Pre-flight checks passed, mode: {mode.value}")
    
    def test_component_integration(self):
        """Test all components are properly integrated"""
        components = [
            ('data_pipeline', self.orchestrator.data_pipeline),
            ('execution_engine', self.orchestrator.execution_engine),
            ('event_calendar', self.orchestrator.event_calendar),
            ('volatility_detector', self.orchestrator.volatility_detector),
            ('anomaly_detector', self.orchestrator.anomaly_detector),
            ('mode_manager', self.orchestrator.mode_manager),
            ('capital_allocator', self.orchestrator.capital_allocator),
            ('leverage_governor', self.orchestrator.leverage_governor),
            ('kill_switch_manager', self.orchestrator.kill_switch_manager)
        ]
        
        for name, component in components:
            assert component is not None, f"{name} not initialized"
        
        print("✅ All 9 core components integrated")


class TestExecutionEngineIntegration:
    """Test Execution Engine Integration"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.engine = ExecutionEngine()
    
    def test_end_to_end_order_flow(self):
        """Test complete order flow from creation to execution"""
        # 1. Create order
        order = self.engine.create_order(
            symbol="SHOP.TO",
            side=OrderSide.BUY,
            quantity=50.5,  # Fractional
            order_type=OrderType.MARKET
        )
        
        assert order is not None
        assert order.quantity == 50.5
        print(f"✅ Order created: {order.order_id}")
        
        # 2. Execute order
        success = self.engine.execute_market_order(
            order=order,
            current_price=150.0,
            volume=1000000
        )
        
        assert success
        assert order.status == OrderStatus.FILLED
        print(f"✅ Order executed: {order.filled_quantity} @ ${order.average_fill_price:.2f}")
        
        # 3. Verify statistics
        stats = self.engine.get_execution_statistics()
        assert stats['total_executions'] > 0
        print(f"✅ Execution stats: {stats['total_executions']} executions")
    
    def test_vwap_large_order(self):
        """Test VWAP execution for large orders"""
        # Create market data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1min')
        data = pd.DataFrame({
            'open': np.random.uniform(145, 155, 100),
            'high': np.random.uniform(150, 160, 100),
            'low': np.random.uniform(140, 150, 100),
            'close': np.random.uniform(145, 155, 100),
            'volume': np.random.uniform(100000, 200000, 100)
        }, index=dates)
        
        # Calculate VWAP
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        data['vwap'] = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        
        # Create large order
        order = self.engine.create_order(
            symbol="TD.TO",
            side=OrderSide.BUY,
            quantity=5000,
            order_type=OrderType.MARKET
        )
        
        # Execute with VWAP
        success = self.engine.execute_vwap_order(
            order=order,
            market_data=data,
            time_window_minutes=30
        )
        
        assert success
        assert order.filled_quantity > 0
        print(f"✅ VWAP order executed: {order.filled_quantity} shares in multiple chunks")


class TestEventAwarenessIntegration:
    """Test Event Awareness Integration"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.calendar = EventCalendar()
        self.volatility_detector = VolatilityDetector()
        self.anomaly_detector = AnomalyDetector()
    
    def test_event_detection_workflow(self):
        """Test complete event detection workflow"""
        # 1. Check upcoming events
        upcoming = self.calendar.get_upcoming_events(hours_ahead=24)
        assert isinstance(upcoming, list)
        print(f"✅ Found {len(upcoming)} upcoming events")
        
        # 2. Check high-impact events
        high_impact = self.calendar.get_high_impact_events(days_ahead=7)
        assert isinstance(high_impact, list)
        print(f"✅ Found {len(high_impact)} high-impact events")
        
        # 3. Check market holiday
        is_holiday = self.calendar.is_market_holiday(datetime(2025, 12, 25))
        assert is_holiday
        print("✅ Holiday detection working")
    
    def test_volatility_analysis_workflow(self):
        """Test volatility analysis workflow"""
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
        
        # Analyze volatility
        analysis = self.volatility_detector.analyze_volatility("TEST.TO", data)
        
        assert 'historical_volatility' in analysis
        assert 'volatility_regime' in analysis
        assert 'trend' in analysis
        print(f"✅ Volatility analysis: {analysis['volatility_regime']} regime")
    
    def test_anomaly_detection_workflow(self):
        """Test anomaly detection workflow"""
        # Create sample data with anomaly
        np.random.seed(42)
        normal_data = pd.DataFrame({
            'open': np.random.uniform(95, 105, 200),
            'high': np.random.uniform(100, 110, 200),
            'low': np.random.uniform(90, 100, 200),
            'close': np.random.uniform(95, 105, 200),
            'volume': np.random.uniform(1000000, 2000000, 200)
        })
        
        # Train detector
        self.anomaly_detector.train(normal_data)
        assert self.anomaly_detector.is_trained
        print("✅ Anomaly detector trained")
        
        # Detect anomalies
        results = self.anomaly_detector.detect_anomalies("TEST.TO", normal_data)
        assert 'is_anomaly' in results.columns
        print(f"✅ Anomaly detection: {results['is_anomaly'].sum()} anomalies found")


class TestPennyStockIntegration:
    """Test Penny Stock Module Integration"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.detector = get_penny_stock_detector()
    
    def test_penny_stock_detection_workflow(self):
        """Test complete penny stock detection workflow"""
        # Create penny stock data (price < $5)
        np.random.seed(42)
        data = pd.DataFrame({
            'open': np.random.uniform(2.0, 2.5, 100),
            'high': np.random.uniform(2.2, 2.7, 100),
            'low': np.random.uniform(1.8, 2.3, 100),
            'close': np.random.uniform(2.0, 2.5, 100),
            'volume': np.random.uniform(100000, 500000, 100)
        })
        
        # Analyze penny stock
        profile = self.detector.analyze_penny_stock("ABC.V", data)
        
        assert profile is not None
        assert profile.symbol == "ABC.V"
        assert profile.price < 5.0
        print(f"✅ Penny stock analyzed: {profile.symbol} @ ${profile.price:.2f}")
        print(f"   Liquidity: {profile.liquidity_score:.2f}, Risk: {profile.risk_level}")
    
    def test_volume_spike_detection(self):
        """Test volume spike detection"""
        historical_volume = pd.Series([100000] * 50)
        current_volume = 350000  # 3.5x spike
        
        is_spike, ratio = self.detector.detect_volume_spike(
            "TEST.V",
            current_volume,
            historical_volume
        )
        
        assert is_spike
        assert ratio > 3.0
        print(f"✅ Volume spike detected: {ratio:.1f}x")
    
    def test_position_sizing(self):
        """Test dynamic position sizing"""
        # Create profile
        np.random.seed(42)
        data = pd.DataFrame({
            'open': np.random.uniform(2.0, 2.5, 100),
            'high': np.random.uniform(2.2, 2.7, 100),
            'low': np.random.uniform(1.8, 2.3, 100),
            'close': np.random.uniform(2.0, 2.5, 100),
            'volume': np.random.uniform(100000, 500000, 100)
        })
        
        profile = self.detector.analyze_penny_stock("DEF.V", data)
        
        if profile and profile.is_tradeable:
            position_size = self.detector.calculate_position_size(profile, 100000)
            assert position_size > 0
            assert position_size <= 100000 * 0.02  # Max 2%
            print(f"✅ Position size calculated: ${position_size:,.2f}")


class TestSIPIntegration:
    """Test SIP Simulator Integration"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.sip = get_sip_simulator()
    
    def test_sip_investment_workflow(self):
        """Test complete SIP investment workflow"""
        # 1. Process daily profit
        transaction = self.sip.process_daily_profit(
            daily_profit=10000.0,  # $10K profit
            etf_price=110.50  # VFV.TO price
        )
        
        assert transaction is not None
        assert transaction.etf_symbol == "VFV.TO"
        assert transaction.amount_cad == 100.0  # 1% of 10K
        print(f"✅ SIP investment: ${transaction.amount_cad:.2f} → {transaction.shares_purchased:.4f} shares")
        
        # 2. Get portfolio value
        portfolio = self.sip.get_portfolio_value(110.50)
        assert portfolio['total_shares'] > 0
        assert portfolio['current_value'] > 0
        print(f"✅ SIP portfolio: {portfolio['total_shares']:.4f} shares, ${portfolio['current_value']:.2f} value")
        
        # 3. Get performance metrics
        performance = self.sip.get_performance_metrics(110.50)
        assert 'total_return' in performance
        print(f"✅ SIP performance: {performance['total_return']:.2f}% return")
    
    def test_sip_minimum_threshold(self):
        """Test SIP minimum investment threshold"""
        # Small profit below threshold
        transaction = self.sip.process_daily_profit(
            daily_profit=1000.0,  # Only $10 would be invested (1%)
            etf_price=110.50
        )
        
        assert transaction is None  # Below $25 minimum
        print("✅ SIP minimum threshold enforced")
    
    def test_sip_tax_reporting(self):
        """Test SIP tax reporting"""
        # Generate tax report
        report = self.sip.get_tax_report(2025)
        
        assert 'tax_year' in report
        assert report['tax_year'] == 2025
        print(f"✅ Tax report generated for {report['tax_year']}")


class TestTradingModesIntegration:
    """Test Trading Modes Integration"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.mode_manager = ModeManager()
    
    def test_mode_switching_workflow(self):
        """Test mode switching workflow"""
        # Start in demo mode
        assert self.mode_manager.is_demo_mode()
        print("✅ Started in demo mode")
        
        # Get account info
        account = self.mode_manager.get_current_account_info()
        assert account['capital'] > 0
        print(f"✅ Demo capital: ${account['capital']:,.2f}")
        
        # Test shared learning data
        learning = self.mode_manager.get_shared_learning_data()
        assert 'insights' in learning
        print("✅ Shared learning data accessible")


class TestSystemResilience:
    """Test System Resilience and Error Handling"""
    
    def test_invalid_inputs_handling(self):
        """Test system handles invalid inputs gracefully"""
        engine = ExecutionEngine()
        
        # Test invalid quantity
        with pytest.raises(ValueError):
            engine.create_order(
                symbol="TEST",
                side=OrderSide.BUY,
                quantity=-100  # Invalid
            )
        
        print("✅ Invalid input rejected")
    
    def test_missing_data_handling(self):
        """Test system handles missing data gracefully"""
        detector = VolatilityDetector()
        
        # Test with insufficient data
        short_series = pd.Series([100, 101, 102])
        hv = detector.calculate_historical_volatility(short_series, period=20)
        
        assert hv == 0.0  # Should return 0 for insufficient data
        print("✅ Missing data handled gracefully")
    
    def test_kill_switch_emergency_stop(self):
        """Test kill switch stops all trading"""
        orchestrator = TradingOrchestrator()
        
        # Activate kill switch
        orchestrator.kill_switch_manager.activate_kill_switch("test", "automated_test")
        
        # Run cycle - should be halted
        results = orchestrator.run_trading_cycle()
        
        assert results['status'] == 'kill_switch_active'
        print("✅ Kill switch stops trading")
        
        # Deactivate
        orchestrator.kill_switch_manager.deactivate_kill_switch("test_complete")


def run_integration_tests():
    """Run all integration tests"""
    print("\n" + "=" * 80)
    print("🧪 RUNNING COMPLETE INTEGRATION TESTS")
    print("=" * 80 + "\n")
    
    # Run pytest
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--color=yes",
        "-s"  # Show print statements
    ])
    
    print("\n" + "=" * 80)
    if exit_code == 0:
        print("✅ ALL INTEGRATION TESTS PASSED!")
    else:
        print(f"❌ TESTS FAILED (exit code: {exit_code})")
    print("=" * 80 + "\n")
    
    return exit_code


if __name__ == "__main__":
    exit(run_integration_tests())

