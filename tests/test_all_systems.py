"""
Comprehensive System Test

Tests all major components:
- Execution Engine
- Event Awareness (Calendar, Volatility, Anomaly Detection)
- AI Model Stack
- RL Core
- Trading Modes
- Reporting System
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
    VolatilityDetector, VolatilityRegime,
    AnomalyDetector, AnomalyType
)
from src.ai.model_stack import LSTMModel, GRUTransformerModel, MetaEnsemble
from src.ai.rl import TradingEnvironment, PPOAgent, DQNAgent
from src.trading_modes import ModeManager
from src.reporting import ReportGenerator, ReportScheduler


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
    
    def test_fractional_shares(self):
        """Test fractional share support"""
        order = self.engine.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10.5,
            order_type=OrderType.MARKET
        )
        
        assert order.quantity == 10.5
    
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
    
    def test_market_holiday(self):
        """Test holiday detection"""
        # The calendar should have Canadian holidays loaded
        # Check if Christmas 2025 is a holiday
        christmas = datetime(2025, 12, 25)
        is_holiday = self.calendar.is_market_holiday(christmas)
        
        assert is_holiday


class TestVolatilityDetector:
    """Test Volatility Detector"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.detector = VolatilityDetector(lookback_period=20)
    
    def test_historical_volatility(self):
        """Test historical volatility calculation"""
        # Create sample price data
        prices = pd.Series(np.random.uniform(95, 105, 100).cumsum())
        
        hv = self.detector.calculate_historical_volatility(prices, period=20)
        
        assert hv > 0
        assert hv < 200  # Reasonable volatility range
    
    def test_atr_calculation(self):
        """Test ATR calculation"""
        # Create sample OHLC data
        data = pd.DataFrame({
            'open': np.random.uniform(95, 105, 100),
            'high': np.random.uniform(100, 110, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(95, 105, 100)
        })
        
        atr = self.detector.calculate_atr(data, period=14)
        
        assert atr > 0
    
    def test_volatility_regime_classification(self):
        """Test regime classification"""
        # Test different volatility levels
        assert self.detector.classify_volatility_regime(8.0) == VolatilityRegime.VERY_LOW
        assert self.detector.classify_volatility_regime(12.0) == VolatilityRegime.LOW
        assert self.detector.classify_volatility_regime(20.0) == VolatilityRegime.NORMAL
        assert self.detector.classify_volatility_regime(35.0) == VolatilityRegime.HIGH
        assert self.detector.classify_volatility_regime(50.0) == VolatilityRegime.EXTREME
    
    def test_volatility_spike_detection(self):
        """Test spike detection"""
        # Create historical data with a spike
        historical = pd.Series([10, 11, 10, 12, 11, 10, 11, 12, 10, 11] * 5)
        current = 30  # Spike
        
        is_spike, z_score = self.detector.detect_volatility_spike(
            symbol="TEST",
            current_volatility=current,
            historical_data=historical
        )
        
        assert is_spike
        assert z_score > 2.0


class TestAnomalyDetector:
    """Test Anomaly Detector"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.detector = AnomalyDetector(contamination=0.05)
    
    def test_train_model(self):
        """Test model training"""
        # Create sample data
        data = pd.DataFrame({
            'open': np.random.uniform(95, 105, 200),
            'high': np.random.uniform(100, 110, 200),
            'low': np.random.uniform(90, 100, 200),
            'close': np.random.uniform(95, 105, 200),
            'volume': np.random.uniform(1000000, 2000000, 200)
        })
        
        self.detector.train(data)
        
        assert self.detector.is_trained
    
    def test_detect_volume_anomaly(self):
        """Test volume anomaly detection"""
        historical_volume = pd.Series([1000000, 1100000, 900000, 1050000, 980000] * 10)
        current_volume = 5000000  # Anomaly
        
        is_anomaly, z_score = self.detector.detect_volume_anomaly(
            symbol="TEST",
            current_volume=current_volume,
            historical_volume=historical_volume
        )
        
        assert is_anomaly
        assert z_score > 3.0
    
    def test_detect_price_anomaly(self):
        """Test price anomaly detection"""
        historical_prices = pd.Series(np.linspace(100, 105, 50))
        current_price = 150  # Anomaly
        
        is_anomaly, z_score = self.detector.detect_price_anomaly(
            symbol="TEST",
            current_price=current_price,
            historical_prices=historical_prices
        )
        
        assert is_anomaly


class TestAIModelStack:
    """Test AI Model Stack"""
    
    def test_lstm_model(self):
        """Test LSTM model"""
        model = LSTMModel(
            input_size=10,
            hidden_size=64,
            num_layers=2,
            output_size=1
        )
        
        # Create sample data
        X = np.random.randn(100, 60, 10).astype(np.float32)
        y = np.random.randn(100, 1).astype(np.float32)
        
        # Train model
        loss = model.train(X, y, epochs=2, batch_size=32)
        
        assert loss > 0
    
    def test_gru_transformer_model(self):
        """Test GRU/Transformer model"""
        model = GRUTransformerModel(
            input_size=20,
            hidden_size=64,
            num_layers=2,
            output_size=1
        )
        
        # Create sample data
        X = np.random.randn(100, 60, 20).astype(np.float32)
        y = np.random.randn(100, 1).astype(np.float32)
        
        # Train model
        loss = model.train(X, y, epochs=2, batch_size=32)
        
        assert loss > 0
    
    def test_meta_ensemble(self):
        """Test Meta-ensemble"""
        lstm = LSTMModel(input_size=10, hidden_size=32, num_layers=1, output_size=1)
        gru = GRUTransformerModel(input_size=10, hidden_size=32, num_layers=1, output_size=1)
        
        ensemble = MetaEnsemble(lstm_model=lstm, gru_model=gru)
        
        # Create sample data
        X_lstm = np.random.randn(10, 60, 10).astype(np.float32)
        X_gru = np.random.randn(10, 60, 10).astype(np.float32)
        
        # Get predictions
        predictions = ensemble.predict(X_lstm, X_gru)
        
        assert len(predictions) == 10


class TestTradingModes:
    """Test Trading Modes"""
    
    def test_mode_manager(self):
        """Test mode manager"""
        manager = ModeManager()
        
        # Test initial state
        assert manager.current_mode == "demo_mode"
        
        # Test mode switching
        manager.switch_mode("live_mode")
        assert manager.current_mode == "live_mode"
        
        # Test capital tracking
        assert manager.get_capital() > 0


class TestReportingSystem:
    """Test Reporting System"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.generator = ReportGenerator(output_dir="reports/test")
    
    def test_generate_daily_report(self):
        """Test daily report generation"""
        report = self.generator.generate_daily_report()
        
        assert report is not None
        assert 'date' in report
        assert 'report_type' in report
    
    def test_generate_weekly_report(self):
        """Test weekly report generation"""
        report = self.generator.generate_weekly_report()
        
        assert report is not None
        assert report['report_type'] == 'weekly'
    
    def test_ai_learning_summary(self):
        """Test AI learning summary generation"""
        summary = self.generator._generate_ai_learning_summary()
        
        assert summary is not None
        assert 'training_sessions' in summary


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("🧪 RUNNING COMPREHENSIVE SYSTEM TESTS")
    print("=" * 80)
    
    # Run pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--color=yes"
    ])


if __name__ == "__main__":
    run_all_tests()

