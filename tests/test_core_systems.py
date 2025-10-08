"""
Core Systems Test (No ML Dependencies Required)

Tests core components that don't require ML libraries:
- Execution Engine
- Event Calendar
- Volatility Detector
- Anomaly Detector (basic functions)
- Trading Modes
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import components
from src.execution import ExecutionEngine, Order, OrderType, OrderSide, OrderStatus
from src.event_awareness import (
    EventCalendar, Event, EventType, EventImportance,
    VolatilityDetector, VolatilityRegime
)
from src.trading_modes import ModeManager


class TestExecutionEngine:
    """Test Execution Engine"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.engine = ExecutionEngine(
            commission_rate=0.001,
            slippage_bps=5.0,
            allow_fractional=True
        )
    
    def test_create_order(self):
        """Test order creation"""
        order = self.engine.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )
        
        assert order is not None
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.status == OrderStatus.PENDING
        print(f"✅ Order created: {order.order_id}")
    
    def test_market_order_execution(self):
        """Test market order execution"""
        order = self.engine.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )
        
        # Execute order
        success = self.engine.execute_market_order(
            order=order,
            current_price=150.0,
            volume=1000000
        )
        
        assert success
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 100
        assert order.average_fill_price > 0
        print(f"✅ Market order executed: {order.filled_quantity} @ ${order.average_fill_price:.2f}")
    
    def test_fractional_shares(self):
        """Test fractional share support"""
        order = self.engine.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10.5,
            order_type=OrderType.MARKET
        )
        
        assert order.quantity == 10.5
        print(f"✅ Fractional shares supported: {order.quantity}")
    
    def test_vwap_execution(self):
        """Test VWAP execution"""
        # Create sample market data
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
        
        order = self.engine.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=1000,
            order_type=OrderType.MARKET
        )
        
        success = self.engine.execute_vwap_order(
            order=order,
            market_data=data,
            time_window_minutes=30
        )
        
        assert success
        assert order.filled_quantity > 0
        print(f"✅ VWAP order executed: {order.filled_quantity} @ ${order.average_fill_price:.2f}")
    
    def test_execution_statistics(self):
        """Test execution statistics"""
        # Execute a few orders
        for i in range(3):
            order = self.engine.create_order(
                symbol=f"TEST{i}",
                side=OrderSide.BUY,
                quantity=100,
                order_type=OrderType.MARKET
            )
            self.engine.execute_market_order(order, 100.0 + i, 1000000)
        
        stats = self.engine.get_execution_statistics()
        
        assert stats['total_executions'] == 3
        assert stats['total_volume'] == 300
        print(f"✅ Execution statistics: {stats}")


class TestEventCalendar:
    """Test Event Calendar"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.calendar = EventCalendar(calendar_file="data/test_event_calendar.json")
    
    def test_add_event(self):
        """Test adding events"""
        event = Event(
            event_id="test_event_1",
            title="Test Economic Release",
            event_type=EventType.ECONOMIC,
            scheduled_time=datetime.now() + timedelta(days=1),
            importance=EventImportance.HIGH
        )
        
        self.calendar.add_event(event)
        
        retrieved = self.calendar.get_event("test_event_1")
        assert retrieved is not None
        assert retrieved.title == "Test Economic Release"
        print(f"✅ Event added and retrieved: {retrieved.title}")
    
    def test_get_upcoming_events(self):
        """Test getting upcoming events"""
        # Add future event
        event = Event(
            event_id="test_future_1",
            title="Future Event",
            event_type=EventType.ECONOMIC,
            scheduled_time=datetime.now() + timedelta(hours=12),
            importance=EventImportance.MEDIUM
        )
        
        self.calendar.add_event(event)
        
        upcoming = self.calendar.get_upcoming_events(hours_ahead=24)
        assert len(upcoming) > 0
        print(f"✅ Found {len(upcoming)} upcoming events")
    
    def test_market_holiday(self):
        """Test holiday detection"""
        # The calendar should have Canadian holidays loaded
        # Check if Christmas 2025 is a holiday
        christmas = datetime(2025, 12, 25)
        is_holiday = self.calendar.is_market_holiday(christmas)
        
        assert is_holiday
        print(f"✅ Christmas 2025 correctly identified as holiday: {is_holiday}")
    
    def test_high_impact_events(self):
        """Test high impact event detection"""
        high_impact = self.calendar.get_high_impact_events(days_ahead=365)
        
        assert len(high_impact) > 0
        print(f"✅ Found {len(high_impact)} high impact events in next year")


class TestVolatilityDetector:
    """Test Volatility Detector"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.detector = VolatilityDetector(lookback_period=20)
    
    def test_historical_volatility(self):
        """Test historical volatility calculation"""
        # Create sample price data
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5))
        
        hv = self.detector.calculate_historical_volatility(prices, period=20)
        
        assert hv > 0
        assert hv < 200  # Reasonable volatility range
        print(f"✅ Historical volatility calculated: {hv:.2f}%")
    
    def test_atr_calculation(self):
        """Test ATR calculation"""
        # Create sample OHLC data
        np.random.seed(42)
        base_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        data = pd.DataFrame({
            'open': base_prices + np.random.randn(100) * 0.1,
            'high': base_prices + np.abs(np.random.randn(100) * 0.5),
            'low': base_prices - np.abs(np.random.randn(100) * 0.5),
            'close': base_prices + np.random.randn(100) * 0.2
        })
        
        atr = self.detector.calculate_atr(data, period=14)
        
        assert atr > 0
        print(f"✅ ATR calculated: {atr:.4f}")
    
    def test_volatility_regime_classification(self):
        """Test regime classification"""
        # Test different volatility levels
        assert self.detector.classify_volatility_regime(8.0) == VolatilityRegime.VERY_LOW
        assert self.detector.classify_volatility_regime(12.0) == VolatilityRegime.LOW
        assert self.detector.classify_volatility_regime(20.0) == VolatilityRegime.NORMAL
        assert self.detector.classify_volatility_regime(35.0) == VolatilityRegime.HIGH
        assert self.detector.classify_volatility_regime(50.0) == VolatilityRegime.EXTREME
        print(f"✅ All volatility regimes correctly classified")
    
    def test_volatility_analysis(self):
        """Test comprehensive volatility analysis"""
        np.random.seed(42)
        base_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        data = pd.DataFrame({
            'open': base_prices + np.random.randn(100) * 0.1,
            'high': base_prices + np.abs(np.random.randn(100) * 0.5),
            'low': base_prices - np.abs(np.random.randn(100) * 0.5),
            'close': base_prices + np.random.randn(100) * 0.2
        })
        
        analysis = self.detector.analyze_volatility("TEST", data)
        
        assert 'historical_volatility' in analysis
        assert 'volatility_regime' in analysis
        assert 'trend' in analysis
        print(f"✅ Volatility analysis: HV={analysis['historical_volatility']:.2f}%, Regime={analysis['volatility_regime']}")


class TestTradingModes:
    """Test Trading Modes"""
    
    def test_mode_manager(self):
        """Test mode manager"""
        manager = ModeManager()
        
        # Test initial state (should be demo mode by default)
        assert manager.is_demo_mode()
        print(f"✅ Mode manager initialized: {manager.current_mode.value}")
        
        # Test getting account info
        account_info = manager.get_current_account_info()
        assert 'capital' in account_info
        assert 'num_trades' in account_info
        assert account_info['capital'] > 0
        print(f"✅ Account info retrieved: ${account_info['capital']:,.2f} with {account_info['num_trades']} trades")
        
        # Test shared learning data
        learning_data = manager.get_shared_learning_data()
        assert 'insights' in learning_data
        assert 'total_trades' in learning_data
        print(f"✅ Shared learning data accessible: {learning_data['total_trades']} total trades")


def run_tests():
    """Run all tests"""
    print("=" * 80)
    print("🧪 RUNNING CORE SYSTEM TESTS (No ML Dependencies)")
    print("=" * 80)
    print()
    
    # Run pytest
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--color=yes",
        "-s"  # Show print statements
    ])
    
    print()
    print("=" * 80)
    if exit_code == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ TESTS FAILED (exit code: {exit_code})")
    print("=" * 80)
    
    return exit_code


if __name__ == "__main__":
    exit(run_tests())

