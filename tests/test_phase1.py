"""
Phase 1 Test Suite - Core Trading Infrastructure

This module tests all Phase 1 components:
- Mode Manager
- Position Management
- Exit Strategy Engine
- Risk Management
- Order Execution Engine
"""

import pytest
import tempfile
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.mode_manager import ModeManager, get_mode_manager, set_mode, get_current_mode
from config.database import DatabaseManager, get_database_manager
from trading.positions import PositionManager, get_position_manager, track_position, get_open_positions
from trading.exit_strategy import ExitStrategyEngine, get_exit_strategy_engine, generate_sell_signals
from trading.risk import RiskManager, get_risk_manager, calculate_position_size, check_portfolio_limits
from trading.execution import ExecutionEngine, get_execution_engine, execute_buy_order, OrderType

class TestModeManager:
    """Test Mode Manager functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create temporary config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "mode_config.json")
        
        # Initialize mode manager with temp config
        self.mode_manager = ModeManager(self.config_path)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_mode_manager_initialization(self):
        """Test mode manager initialization."""
        assert self.mode_manager is not None
        assert self.mode_manager.get_current_mode() == "DEMO"  # Default mode
    
    def test_mode_switching(self):
        """Test mode switching functionality."""
        # Test switching to LIVE mode
        result = self.mode_manager.set_mode("LIVE")
        assert result is True
        assert self.mode_manager.get_current_mode() == "LIVE"
        
        # Test switching back to DEMO mode
        result = self.mode_manager.set_mode("DEMO")
        assert result is True
        assert self.mode_manager.get_current_mode() == "DEMO"
    
    def test_invalid_mode(self):
        """Test invalid mode handling."""
        result = self.mode_manager.set_mode("INVALID")
        assert result is False
        assert self.mode_manager.get_current_mode() == "DEMO"  # Should remain unchanged
    
    def test_mode_config(self):
        """Test mode configuration retrieval."""
        config = self.mode_manager.get_mode_config()
        assert config.mode == "DEMO"
        assert config.order_execution == "PAPER"
        assert config.balance_tracking is True
    
    def test_mode_validation(self):
        """Test mode validation."""
        assert self.mode_manager.validate_mode() is True
        
        # Test with invalid mode
        self.mode_manager.current_mode = "INVALID"
        assert self.mode_manager.validate_mode() is False
    
    def test_mode_info(self):
        """Test mode information retrieval."""
        info = self.mode_manager.get_mode_info()
        assert info["current_mode"] == "DEMO"
        assert info["is_demo"] is True
        assert info["is_live"] is False
        assert info["is_paper_trading"] is True
        assert info["is_real_trading"] is False

class TestPositionManager:
    """Test Position Manager functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        # Mock the mode manager to use DEMO mode
        with patch('trading.positions.get_current_mode', return_value='DEMO'):
            with patch('trading.positions.get_connection') as mock_conn:
                # Mock database connection
                mock_conn.return_value.__enter__.return_value.cursor.return_value.lastrowid = 1
                self.position_manager = PositionManager()
    
    def test_position_tracking(self):
        """Test position tracking functionality."""
        with patch('trading.positions.get_current_mode', return_value='DEMO'):
            with patch('trading.positions.get_connection') as mock_conn:
                mock_cursor = MagicMock()
                mock_cursor.lastrowid = 1
                mock_conn.return_value.__enter__.return_value.cursor.return_value = mock_cursor
                
                position = track_position("AAPL", 150.0, 10, stop_loss=140.0, take_profit=160.0)
                
                assert position.symbol == "AAPL"
                assert position.entry_price == 150.0
                assert position.quantity == 10
                assert position.stop_loss == 140.0
                assert position.take_profit == 160.0
                assert position.status == "OPEN"
                assert position.mode == "DEMO"
    
    def test_position_validation(self):
        """Test position data validation."""
        with patch('trading.positions.get_current_mode', return_value='DEMO'):
            # Test invalid symbol
            with pytest.raises(ValueError):
                track_position("", 150.0, 10)
            
            # Test invalid price
            with pytest.raises(ValueError):
                track_position("AAPL", -150.0, 10)
            
            # Test invalid quantity
            with pytest.raises(ValueError):
                track_position("AAPL", 150.0, -10)
    
    def test_pnl_calculation(self):
        """Test P&L calculation."""
        with patch('trading.positions.get_current_mode', return_value='DEMO'):
            with patch('trading.positions.get_position_by_symbol') as mock_get_pos:
                # Mock position
                mock_position = MagicMock()
                mock_position.entry_price = 150.0
                mock_position.quantity = 10
                mock_get_pos.return_value = mock_position
                
                pnl = self.position_manager.calculate_pnl("AAPL", 160.0)
                assert pnl == 100.0  # (160 - 150) * 10
                
                pnl_percent = self.position_manager.calculate_pnl_percent("AAPL", 160.0)
                assert pnl_percent == 6.67  # ((160 - 150) / 150) * 100

class TestExitStrategyEngine:
    """Test Exit Strategy Engine functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        with patch('trading.exit_strategy.get_current_mode', return_value='DEMO'):
            self.exit_engine = ExitStrategyEngine()
    
    def test_stop_loss_check(self):
        """Test stop loss checking."""
        from trading.positions import Position
        
        # Create mock position
        position = Position(
            id=1,
            symbol="AAPL",
            entry_price=150.0,
            quantity=10,
            entry_time=datetime.now(),
            exit_price=None,
            exit_time=None,
            status="OPEN",
            stop_loss=140.0,
            take_profit=160.0,
            pnl=0.0,
            pnl_percent=0.0,
            mode="DEMO",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Test stop loss trigger
        signal = self.exit_engine.check_stop_loss(position, "DEMO", 135.0)
        assert signal is not None
        assert signal.exit_reason.value == "STOP_LOSS"
        assert signal.exit_price == 140.0
        
        # Test no stop loss trigger
        signal = self.exit_engine.check_stop_loss(position, "DEMO", 145.0)
        assert signal is None
    
    def test_take_profit_check(self):
        """Test take profit checking."""
        from trading.positions import Position
        
        # Create mock position
        position = Position(
            id=1,
            symbol="AAPL",
            entry_price=150.0,
            quantity=10,
            entry_time=datetime.now(),
            exit_price=None,
            exit_time=None,
            status="OPEN",
            stop_loss=140.0,
            take_profit=160.0,
            pnl=0.0,
            pnl_percent=0.0,
            mode="DEMO",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Test take profit trigger
        signal = self.exit_engine.check_take_profit(position, "DEMO", 165.0)
        assert signal is not None
        assert signal.exit_reason.value == "TAKE_PROFIT"
        assert signal.exit_price == 160.0
        
        # Test no take profit trigger
        signal = self.exit_engine.check_take_profit(position, "DEMO", 155.0)
        assert signal is None
    
    def test_time_based_exit(self):
        """Test time-based exit checking."""
        from trading.positions import Position
        
        # Create mock position with old entry time
        old_time = datetime.now() - timedelta(days=35)  # 35 days ago
        position = Position(
            id=1,
            symbol="AAPL",
            entry_price=150.0,
            quantity=10,
            entry_time=old_time,
            exit_price=None,
            exit_time=None,
            status="OPEN",
            stop_loss=140.0,
            take_profit=160.0,
            pnl=0.0,
            pnl_percent=0.0,
            mode="DEMO",
            created_at=old_time,
            updated_at=old_time
        )
        
        # Test time-based exit trigger
        signal = self.exit_engine.check_time_based_exit(position, "DEMO")
        assert signal is not None
        assert signal.exit_reason.value == "TIME_BASED"

class TestRiskManager:
    """Test Risk Manager functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        with patch('trading.risk.get_current_mode', return_value='DEMO'):
            with patch('trading.risk.get_risk_limits', return_value={
                'max_position_size': 0.10,
                'max_daily_drawdown': 0.05,
                'max_portfolio_risk': 0.20,
                'min_confidence': 0.70,
                'max_positions': 10
            }):
                self.risk_manager = RiskManager()
    
    def test_position_sizing(self):
        """Test position sizing calculation."""
        risk_metrics = calculate_position_size(
            signal_confidence=0.8,
            account_balance=10000.0,
            volatility=0.02,
            entry_price=150.0,
            stop_loss=140.0
        )
        
        assert risk_metrics.position_size > 0
        assert risk_metrics.risk_amount > 0
        assert risk_metrics.stop_loss == 140.0
        assert risk_metrics.take_profit > 150.0  # Should be above entry price
    
    def test_portfolio_limits(self):
        """Test portfolio limit checking."""
        with patch('trading.risk.get_open_positions', return_value=[]):
            limits = check_portfolio_limits()
            
            assert limits["mode"] == "DEMO"
            assert limits["total_positions"] == 0
            assert limits["can_open_new_position"] is True
            assert limits["positions_within_limit"] is True
    
    def test_stop_loss_calculation(self):
        """Test stop loss calculation."""
        stop_loss = self.risk_manager.calculate_stop_loss(150.0, 2.0, 2.0)
        assert stop_loss == 146.0  # 150 - (2.0 * 2.0)
        
        # Test with zero ATR
        stop_loss = self.risk_manager.calculate_stop_loss(150.0, 0.0, 2.0)
        assert stop_loss == 148.5  # 150 * 0.99 (minimum 1% below entry)
    
    def test_take_profit_calculation(self):
        """Test take profit calculation."""
        take_profit = self.risk_manager.calculate_take_profit(150.0, 140.0, 1.5)
        assert take_profit == 165.0  # 150 + (10 * 1.5)

class TestExecutionEngine:
    """Test Execution Engine functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        with patch('trading.execution.get_current_mode', return_value='DEMO'):
            with patch('trading.execution.is_paper_trading', return_value=True):
                with patch('trading.execution.get_connection') as mock_conn:
                    mock_cursor = MagicMock()
                    mock_cursor.lastrowid = 1
                    mock_conn.return_value.__enter__.return_value.cursor.return_value = mock_cursor
                    self.execution_engine = ExecutionEngine()
    
    def test_buy_order_execution(self):
        """Test buy order execution."""
        with patch('trading.execution.get_current_mode', return_value='DEMO'):
            with patch('trading.execution.is_paper_trading', return_value=True):
                with patch('trading.execution.get_connection') as mock_conn:
                    mock_cursor = MagicMock()
                    mock_cursor.lastrowid = 1
                    mock_conn.return_value.__enter__.return_value.cursor.return_value = mock_cursor
                    
                    with patch.object(self.execution_engine, '_validate_buy_order', return_value=True):
                        result = execute_buy_order("AAPL", 10, 150.0)
                        
                        assert result.success is True
                        assert result.order is not None
                        assert result.order.symbol == "AAPL"
                        assert result.order.quantity == 10
                        assert result.order.price == 150.0
                        assert result.mode == "DEMO"
    
    def test_order_validation(self):
        """Test order validation."""
        with patch('trading.execution.get_current_mode', return_value='DEMO'):
            with patch('trading.execution.is_paper_trading', return_value=True):
                # Test invalid symbol
                result = execute_buy_order("", 10, 150.0)
                assert result.success is False
                assert "validation failed" in result.error_message.lower()
                
                # Test invalid quantity
                result = execute_buy_order("AAPL", -10, 150.0)
                assert result.success is False
                
                # Test invalid price
                result = execute_buy_order("AAPL", 10, -150.0)
                assert result.success is False

class TestIntegration:
    """Test integration between Phase 1 components."""
    
    def setup_method(self):
        """Set up test environment."""
        # Mock all external dependencies
        with patch('config.mode_manager.get_current_mode', return_value='DEMO'):
            with patch('trading.positions.get_current_mode', return_value='DEMO'):
                with patch('trading.exit_strategy.get_current_mode', return_value='DEMO'):
                    with patch('trading.risk.get_current_mode', return_value='DEMO'):
                        with patch('trading.execution.get_current_mode', return_value='DEMO'):
                            with patch('trading.execution.is_paper_trading', return_value=True):
                                with patch('trading.execution.get_connection') as mock_conn:
                                    mock_cursor = MagicMock()
                                    mock_cursor.lastrowid = 1
                                    mock_conn.return_value.__enter__.return_value.cursor.return_value = mock_cursor
                                    
                                    # Initialize components
                                    self.mode_manager = get_mode_manager()
                                    self.position_manager = get_position_manager()
                                    self.exit_engine = get_exit_strategy_engine()
                                    self.risk_manager = get_risk_manager()
                                    self.execution_engine = get_execution_engine()
    
    def test_complete_trading_cycle(self):
        """Test complete trading cycle integration."""
        # This would test the full cycle:
        # 1. Mode switching
        # 2. Position tracking
        # 3. Risk management
        # 4. Order execution
        # 5. Exit strategy
        
        # For now, just test that all components can be initialized together
        assert self.mode_manager is not None
        assert self.position_manager is not None
        assert self.exit_engine is not None
        assert self.risk_manager is not None
        assert self.execution_engine is not None
        
        # Test mode consistency
        assert self.mode_manager.get_current_mode() == "DEMO"
    
    def test_mode_isolation(self):
        """Test that LIVE and DEMO modes are properly isolated."""
        # This would test that data doesn't leak between modes
        # For now, just verify mode manager functionality
        
        # Switch to LIVE mode
        result = self.mode_manager.set_mode("LIVE")
        assert result is True
        assert self.mode_manager.get_current_mode() == "LIVE"
        
        # Switch back to DEMO mode
        result = self.mode_manager.set_mode("DEMO")
        assert result is True
        assert self.mode_manager.get_current_mode() == "DEMO"

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
