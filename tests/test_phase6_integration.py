"""
Phase 6 Integration Tests - Drawdown-Aware Kelly & ATR Brackets

Tests the integration of:
1. Drawdown-aware Kelly sizing
2. ATR-based bracket parameters
3. Risk fraction validation
4. Database schema updates
"""

import pytest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trading.risk import RiskManager, RiskMetrics
from trading.atr_brackets import ATRBracketManager, BracketParameters
from trading.execution import ExecutionEngine, Order, OrderType, OrderSide
from config.database import get_database_manager
from config.mode_manager import get_mode_manager


class TestDrawdownAwareKelly:
    """Test drawdown-aware Kelly sizing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.risk_manager = RiskManager(
            max_position_size=10000,
            max_risk_per_trade=0.02,
            max_daily_risk=0.05,
            min_drawdown_scale=0.1,
            drawdown_window_hours=24
        )
        
        # Mock mode manager
        with patch('config.mode_manager.get_mode_manager') as mock_mode:
            mock_mode.return_value.get_mode.return_value = "DEMO"
            self.mode_manager = get_mode_manager()
    
    def test_drawdown_calculation(self):
        """Test daily drawdown calculation."""
        # Mock portfolio snapshots
        with patch('config.database.get_portfolio_snapshots') as mock_snapshots:
            mock_snapshots.return_value = [
                {'portfolio_value': 10000, 'timestamp': '2024-01-01 09:00:00'},
                {'portfolio_value': 9500, 'timestamp': '2024-01-01 10:00:00'},
                {'portfolio_value': 9000, 'timestamp': '2024-01-01 11:00:00'},
                {'portfolio_value': 9200, 'timestamp': '2024-01-01 12:00:00'},
            ]
            
            drawdown = self.risk_manager.calculate_daily_drawdown("DEMO")
            assert drawdown == 0.10  # 10% drawdown from peak
    
    def test_drawdown_scale_calculation(self):
        """Test drawdown scale calculation."""
        # Test normal drawdown
        scale = self.risk_manager.calculate_drawdown_scale(0.05)  # 5% drawdown
        assert scale == 0.5  # 50% scale
        
        # Test high drawdown
        scale = self.risk_manager.calculate_drawdown_scale(0.15)  # 15% drawdown
        assert scale == 0.1  # Minimum scale
        
        # Test no drawdown
        scale = self.risk_manager.calculate_drawdown_scale(0.0)  # No drawdown
        assert scale == 1.0  # Full scale
    
    def test_kelly_fraction_calculation(self):
        """Test Kelly fraction calculation."""
        # Test with good confidence
        kelly = self.risk_manager.calculate_kelly_fraction(0.7, 0.3)  # 70% win, 30% loss
        expected = (0.7 * 0.3 - 0.3 * 0.7) / 0.3
        assert abs(kelly - expected) < 0.001
        
        # Test with poor confidence
        kelly = self.risk_manager.calculate_kelly_fraction(0.4, 0.6)  # 40% win, 60% loss
        assert kelly < 0  # Negative Kelly (should not trade)
    
    def test_drawdown_aware_position_sizing(self):
        """Test position sizing with drawdown awareness."""
        with patch('config.database.get_portfolio_snapshots') as mock_snapshots:
            mock_snapshots.return_value = [
                {'portfolio_value': 10000, 'timestamp': '2024-01-01 09:00:00'},
                {'portfolio_value': 9000, 'timestamp': '2024-01-01 10:00:00'},
            ]
            
            with patch('adaptive.confidence_calibration.get_confidence_calibrator') as mock_calibrator:
                mock_cal = Mock()
                mock_cal.calibrate_confidence.return_value = 0.7
                mock_calibrator.return_value = mock_cal
                
                metrics = self.risk_manager.calculate_position_size(
                    symbol="AAPL",
                    entry_price=150.0,
                    stop_loss=145.0,
                    model_name="test_model",
                    trade_date="2024-01-01"
                )
                
                assert metrics.drawdown_scale < 1.0  # Should be scaled down
                assert metrics.kelly_fraction > 0  # Should have positive Kelly
                assert metrics.calibrated_confidence == 0.7


class TestATRBrackets:
    """Test ATR-based bracket parameters."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.atr_manager = ATRBracketManager()
    
    def test_atr_bracket_calculation(self):
        """Test ATR bracket calculation."""
        params = self.atr_manager.calculate_atr_brackets(
            symbol="AAPL",
            entry_price=150.0,
            atr=3.0,
            atr_multiplier=2.0,
            bracket_type="conservative"
        )
        
        assert params.stop_loss == 144.0  # 150 - (3.0 * 2.0)
        assert params.take_profit == 156.0  # 150 + (3.0 * 2.0)
        assert params.r_multiple == 1.0  # 1:1 risk/reward
        assert params.volatility_percent == 2.0  # (3.0 / 150.0) * 100
    
    def test_bracket_type_variations(self):
        """Test different bracket types."""
        # Conservative brackets
        params = self.atr_manager.calculate_atr_brackets(
            symbol="AAPL",
            entry_price=150.0,
            atr=3.0,
            atr_multiplier=2.0,
            bracket_type="conservative"
        )
        assert params.r_multiple == 1.0
        
        # Aggressive brackets
        params = self.atr_manager.calculate_atr_brackets(
            symbol="AAPL",
            entry_price=150.0,
            atr=3.0,
            atr_multiplier=2.0,
            bracket_type="aggressive"
        )
        assert params.r_multiple == 2.0  # 2:1 risk/reward
    
    def test_volatility_adjustment(self):
        """Test volatility-based adjustments."""
        # High volatility
        params = self.atr_manager.calculate_atr_brackets(
            symbol="TSLA",
            entry_price=200.0,
            atr=10.0,
            atr_multiplier=2.0,
            bracket_type="conservative"
        )
        assert params.volatility_percent == 5.0  # (10.0 / 200.0) * 100
        
        # Low volatility
        params = self.atr_manager.calculate_atr_brackets(
            symbol="AAPL",
            entry_price=150.0,
            atr=1.5,
            atr_multiplier=2.0,
            bracket_type="conservative"
        )
        assert params.volatility_percent == 1.0  # (1.5 / 150.0) * 100


class TestExecutionIntegration:
    """Test execution engine integration with ATR brackets."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.execution_engine = ExecutionEngine()
        
        # Mock dependencies
        with patch('config.mode_manager.get_mode_manager') as mock_mode:
            mock_mode.return_value.get_mode.return_value = "DEMO"
            self.mode_manager = get_mode_manager()
    
    def test_buy_order_with_brackets(self):
        """Test buy order execution with ATR brackets."""
        with patch('config.database.get_database_manager') as mock_db:
            mock_db_manager = Mock()
            mock_db.return_value = mock_db_manager
            
            with patch('trading.atr_brackets.get_atr_bracket_manager') as mock_atr:
                mock_atr_manager = Mock()
                mock_atr.return_value = mock_atr_manager
                
                # Mock bracket parameters
                bracket_params = BracketParameters(
                    symbol="AAPL",
                    entry_price=150.0,
                    stop_loss=144.0,
                    take_profit=156.0,
                    atr=3.0,
                    atr_multiplier=2.0,
                    r_multiple=1.0,
                    bracket_type="conservative",
                    volatility_percent=2.0
                )
                mock_atr_manager.calculate_atr_brackets.return_value = bracket_params
                
                # Mock database operations
                mock_db_manager.get_connection.return_value.__enter__.return_value.cursor.return_value.lastrowid = 1
                
                order = self.execution_engine.execute_buy_order(
                    symbol="AAPL",
                    quantity=100,
                    price=150.0
                )
                
                assert order.symbol == "AAPL"
                assert order.quantity == 100
                assert order.price == 150.0
                assert order.atr == 3.0
                assert order.atr_multiplier == 2.0
                assert order.r_multiple == 1.0
                assert order.bracket_type == "conservative"
                assert order.volatility_percent == 2.0
    
    def test_atr_value_retrieval(self):
        """Test ATR value retrieval."""
        atr = self.execution_engine._get_atr_value("AAPL", "DEMO")
        assert atr > 0  # Should return positive ATR value
        
        # Test different symbols
        atr_tsla = self.execution_engine._get_atr_value("TSLA", "DEMO")
        atr_aapl = self.execution_engine._get_atr_value("AAPL", "DEMO")
        assert atr_tsla > atr_aapl  # TSLA should have higher ATR


class TestDatabaseSchema:
    """Test database schema updates."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.db_manager = get_database_manager()
    
    def test_orders_table_schema(self):
        """Test that orders table has new columns."""
        with self.db_manager.get_connection("DEMO") as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(orders)")
            columns = [row[1] for row in cursor.fetchall()]
            
            # Check for new columns
            assert "atr" in columns
            assert "atr_multiplier" in columns
            assert "r_multiple" in columns
            assert "bracket_type" in columns
            assert "volatility_percent" in columns
    
    def test_bracket_parameters_table(self):
        """Test bracket parameters table exists."""
        with self.db_manager.get_connection("DEMO") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='bracket_parameters'")
            result = cursor.fetchone()
            assert result is not None
    
    def test_portfolio_snapshots_table(self):
        """Test portfolio snapshots table exists."""
        with self.db_manager.get_connection("DEMO") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio_snapshots'")
            result = cursor.fetchone()
            assert result is not None


class TestRiskFractionValidation:
    """Test risk fraction validation and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.risk_manager = RiskManager(
            max_position_size=10000,
            max_risk_per_trade=0.02,
            max_daily_risk=0.05,
            min_drawdown_scale=0.1,
            drawdown_window_hours=24
        )
    
    def test_negative_kelly_handling(self):
        """Test handling of negative Kelly fractions."""
        with patch('config.database.get_portfolio_snapshots') as mock_snapshots:
            mock_snapshots.return_value = [
                {'portfolio_value': 10000, 'timestamp': '2024-01-01 09:00:00'},
            ]
            
            with patch('adaptive.confidence_calibration.get_confidence_calibrator') as mock_calibrator:
                mock_cal = Mock()
                mock_cal.calibrate_confidence.return_value = 0.3  # Low confidence
                mock_calibrator.return_value = mock_cal
                
                metrics = self.risk_manager.calculate_position_size(
                    symbol="AAPL",
                    entry_price=150.0,
                    stop_loss=145.0,
                    model_name="test_model",
                    trade_date="2024-01-01"
                )
                
                # Should fall back to traditional risk sizing
                assert metrics.position_size > 0
                assert metrics.kelly_fraction <= 0  # Negative Kelly
    
    def test_maximum_position_size_cap(self):
        """Test that position size is capped at maximum."""
        with patch('config.database.get_portfolio_snapshots') as mock_snapshots:
            mock_snapshots.return_value = [
                {'portfolio_value': 100000, 'timestamp': '2024-01-01 09:00:00'},
            ]
            
            with patch('adaptive.confidence_calibration.get_confidence_calibrator') as mock_calibrator:
                mock_cal = Mock()
                mock_cal.calibrate_confidence.return_value = 0.95  # Very high confidence
                mock_calibrator.return_value = mock_cal
                
                metrics = self.risk_manager.calculate_position_size(
                    symbol="AAPL",
                    entry_price=150.0,
                    stop_loss=145.0,
                    model_name="test_model",
                    trade_date="2024-01-01"
                )
                
                # Should be capped at max position size
                assert metrics.position_size <= 10000
    
    def test_drawdown_scale_edge_cases(self):
        """Test drawdown scale edge cases."""
        # Test extreme drawdown
        scale = self.risk_manager.calculate_drawdown_scale(0.5)  # 50% drawdown
        assert scale == 0.1  # Should hit minimum
        
        # Test negative drawdown (impossible but test robustness)
        scale = self.risk_manager.calculate_drawdown_scale(-0.1)  # Negative drawdown
        assert scale == 1.0  # Should handle gracefully


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
