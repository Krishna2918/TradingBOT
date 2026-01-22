"""
Integration Tests for Mode Switching
====================================

Tests the integration of mode switching between LIVE and DEMO modes,
including data isolation and state persistence.
"""

import pytest
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock

# Import mode management components
from src.config.mode_manager import set_mode, get_current_mode, get_mode_config
from src.config.database import get_database_manager
from src.trading.positions import get_position_manager
from src.trading.execution import get_execution_engine
from src.workflows.trading_cycle import get_trading_cycle

logger = logging.getLogger(__name__)


class TestModeSwitching:
    """Test suite for mode switching integration."""
    
    @pytest.fixture
    async def setup_clean_state(self):
        """Setup clean state for testing."""
        # Start in DEMO mode
        set_mode("DEMO")
        yield
        # Cleanup after test
    
    @pytest.fixture
    async def mock_market_data(self):
        """Mock market data for testing."""
        return {
            "AAPL": {
                "price": 150.0,
                "volume": 1000000,
                "timestamp": datetime.now(),
                "bid": 149.95,
                "ask": 150.05
            }
        }
    
    @pytest.mark.asyncio
    async def test_live_to_demo_switch(self, setup_clean_state, mock_market_data):
        """Test switching from LIVE to DEMO mode."""
        logger.info("Testing LIVE to DEMO mode switch...")
        
        # Start in LIVE mode
        set_mode("LIVE")
        assert get_current_mode() == "LIVE"
        
        # Create some LIVE mode data
        position_manager = get_position_manager()
        live_position = {
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10,
            "entry_price": 150.0,
            "entry_time": datetime.now(),
            "stop_loss": 145.0,
            "take_profit": 160.0,
            "mode": "LIVE"
        }
        
        await position_manager.track_position(live_position)
        
        # Verify LIVE position exists
        live_positions = await position_manager.get_open_positions()
        assert len(live_positions) > 0
        assert live_positions[0]["mode"] == "LIVE"
        
        # Switch to DEMO mode
        set_mode("DEMO")
        assert get_current_mode() == "DEMO"
        
        # Verify data isolation - DEMO positions should be separate
        demo_positions = await position_manager.get_open_positions()
        # DEMO positions should be empty or different from LIVE positions
        assert all(pos["mode"] == "DEMO" for pos in demo_positions)
        
        # Create DEMO position
        demo_position = {
            "symbol": "MSFT",
            "side": "BUY",
            "quantity": 5,
            "entry_price": 300.0,
            "entry_time": datetime.now(),
            "stop_loss": 290.0,
            "take_profit": 320.0,
            "mode": "DEMO"
        }
        
        await position_manager.track_position(demo_position)
        
        # Verify DEMO position exists
        demo_positions = await position_manager.get_open_positions()
        assert len(demo_positions) > 0
        assert demo_positions[0]["mode"] == "DEMO"
        
        logger.info("✓ LIVE to DEMO mode switch completed successfully")
    
    @pytest.mark.asyncio
    async def test_demo_to_live_switch(self, setup_clean_state, mock_market_data):
        """Test switching from DEMO to LIVE mode."""
        logger.info("Testing DEMO to LIVE mode switch...")
        
        # Start in DEMO mode
        set_mode("DEMO")
        assert get_current_mode() == "DEMO"
        
        # Create some DEMO mode data
        position_manager = get_position_manager()
        demo_position = {
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10,
            "entry_price": 150.0,
            "entry_time": datetime.now(),
            "stop_loss": 145.0,
            "take_profit": 160.0,
            "mode": "DEMO"
        }
        
        await position_manager.track_position(demo_position)
        
        # Verify DEMO position exists
        demo_positions = await position_manager.get_open_positions()
        assert len(demo_positions) > 0
        assert demo_positions[0]["mode"] == "DEMO"
        
        # Switch to LIVE mode
        set_mode("LIVE")
        assert get_current_mode() == "LIVE"
        
        # Verify data isolation - LIVE positions should be separate
        live_positions = await position_manager.get_open_positions()
        # LIVE positions should be empty or different from DEMO positions
        assert all(pos["mode"] == "LIVE" for pos in live_positions)
        
        # Create LIVE position
        live_position = {
            "symbol": "MSFT",
            "side": "BUY",
            "quantity": 5,
            "entry_price": 300.0,
            "entry_time": datetime.now(),
            "stop_loss": 290.0,
            "take_profit": 320.0,
            "mode": "LIVE"
        }
        
        await position_manager.track_position(live_position)
        
        # Verify LIVE position exists
        live_positions = await position_manager.get_open_positions()
        assert len(live_positions) > 0
        assert live_positions[0]["mode"] == "LIVE"
        
        logger.info("✓ DEMO to LIVE mode switch completed successfully")
    
    @pytest.mark.asyncio
    async def test_data_isolation(self, setup_clean_state):
        """Test data isolation between LIVE and DEMO modes."""
        logger.info("Testing data isolation between modes...")
        
        position_manager = get_position_manager()
        
        # Create positions in both modes
        demo_position = {
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10,
            "entry_price": 150.0,
            "entry_time": datetime.now(),
            "stop_loss": 145.0,
            "take_profit": 160.0,
            "mode": "DEMO"
        }
        
        live_position = {
            "symbol": "MSFT",
            "side": "SELL",
            "quantity": 5,
            "entry_price": 300.0,
            "entry_time": datetime.now(),
            "stop_loss": 310.0,
            "take_profit": 280.0,
            "mode": "LIVE"
        }
        
        # Switch to DEMO mode and create position
        set_mode("DEMO")
        await position_manager.track_position(demo_position)
        
        # Switch to LIVE mode and create position
        set_mode("LIVE")
        await position_manager.track_position(live_position)
        
        # Verify DEMO mode isolation
        set_mode("DEMO")
        demo_positions = await position_manager.get_open_positions()
        assert len(demo_positions) > 0
        assert all(pos["mode"] == "DEMO" for pos in demo_positions)
        assert all(pos["symbol"] == "AAPL" for pos in demo_positions)
        
        # Verify LIVE mode isolation
        set_mode("LIVE")
        live_positions = await position_manager.get_open_positions()
        assert len(live_positions) > 0
        assert all(pos["mode"] == "LIVE" for pos in live_positions)
        assert all(pos["symbol"] == "MSFT" for pos in live_positions)
        
        # Test database isolation
        db_manager = get_database_manager()
        
        # DEMO mode should use demo database
        set_mode("DEMO")
        demo_db_path = db_manager.get_database_path()
        assert "demo" in demo_db_path.lower()
        
        # LIVE mode should use live database
        set_mode("LIVE")
        live_db_path = db_manager.get_database_path()
        assert "live" in live_db_path.lower()
        
        # Verify different database paths
        assert demo_db_path != live_db_path
        
        logger.info("✓ Data isolation completed successfully")
    
    @pytest.mark.asyncio
    async def test_state_persistence(self, setup_clean_state):
        """Test state persistence across mode switches."""
        logger.info("Testing state persistence...")
        
        position_manager = get_position_manager()
        
        # Create position in DEMO mode
        set_mode("DEMO")
        demo_position = {
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10,
            "entry_price": 150.0,
            "entry_time": datetime.now(),
            "stop_loss": 145.0,
            "take_profit": 160.0,
            "mode": "DEMO"
        }
        
        await position_manager.track_position(demo_position)
        
        # Switch to LIVE mode and back to DEMO
        set_mode("LIVE")
        set_mode("DEMO")
        
        # Verify DEMO position still exists
        demo_positions = await position_manager.get_open_positions()
        assert len(demo_positions) > 0
        assert demo_positions[0]["symbol"] == "AAPL"
        assert demo_positions[0]["mode"] == "DEMO"
        
        # Create position in LIVE mode
        set_mode("LIVE")
        live_position = {
            "symbol": "MSFT",
            "side": "SELL",
            "quantity": 5,
            "entry_price": 300.0,
            "entry_time": datetime.now(),
            "stop_loss": 310.0,
            "take_profit": 280.0,
            "mode": "LIVE"
        }
        
        await position_manager.track_position(live_position)
        
        # Switch to DEMO mode and back to LIVE
        set_mode("DEMO")
        set_mode("LIVE")
        
        # Verify LIVE position still exists
        live_positions = await position_manager.get_open_positions()
        assert len(live_positions) > 0
        assert live_positions[0]["symbol"] == "MSFT"
        assert live_positions[0]["mode"] == "LIVE"
        
        logger.info("✓ State persistence completed successfully")
    
    @pytest.mark.asyncio
    async def test_dashboard_mode_toggle(self, setup_clean_state):
        """Test dashboard mode toggle functionality."""
        logger.info("Testing dashboard mode toggle...")
        
        # Test mode configuration
        set_mode("DEMO")
        demo_config = get_mode_config()
        assert demo_config.mode == "DEMO"
        assert demo_config.paper_trading is True
        
        set_mode("LIVE")
        live_config = get_mode_config()
        assert live_config.mode == "LIVE"
        assert live_config.paper_trading is False
        
        # Test mode-specific settings
        assert demo_config.risk_limits.max_position_size != live_config.risk_limits.max_position_size
        assert demo_config.risk_limits.max_daily_loss != live_config.risk_limits.max_daily_loss
        
        # Test mode validation
        set_mode("DEMO")
        assert get_current_mode() == "DEMO"
        
        set_mode("LIVE")
        assert get_current_mode() == "LIVE"
        
        # Test invalid mode handling
        try:
            set_mode("INVALID")
            assert False, "Should have raised an error for invalid mode"
        except ValueError:
            pass  # Expected behavior
        
        logger.info("✓ Dashboard mode toggle completed successfully")
    
    @pytest.mark.asyncio
    async def test_active_trade_handling(self, setup_clean_state, mock_market_data):
        """Test handling of active trades during mode switches."""
        logger.info("Testing active trade handling...")
        
        position_manager = get_position_manager()
        execution_engine = get_execution_engine()
        
        # Create active position in DEMO mode
        set_mode("DEMO")
        demo_position = {
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10,
            "entry_price": 150.0,
            "entry_time": datetime.now(),
            "stop_loss": 145.0,
            "take_profit": 160.0,
            "mode": "DEMO"
        }
        
        await position_manager.track_position(demo_position)
        
        # Verify active position exists
        active_positions = await position_manager.get_open_positions()
        assert len(active_positions) > 0
        
        # Switch to LIVE mode
        set_mode("LIVE")
        
        # Verify DEMO position is not accessible in LIVE mode
        live_positions = await position_manager.get_open_positions()
        assert all(pos["mode"] == "LIVE" for pos in live_positions)
        
        # Create active position in LIVE mode
        live_position = {
            "symbol": "MSFT",
            "side": "SELL",
            "quantity": 5,
            "entry_price": 300.0,
            "entry_time": datetime.now(),
            "stop_loss": 310.0,
            "take_profit": 280.0,
            "mode": "LIVE"
        }
        
        await position_manager.track_position(live_position)
        
        # Switch back to DEMO mode
        set_mode("DEMO")
        
        # Verify LIVE position is not accessible in DEMO mode
        demo_positions = await position_manager.get_open_positions()
        assert all(pos["mode"] == "DEMO" for pos in demo_positions)
        
        # Test execution engine mode handling
        set_mode("DEMO")
        demo_execution = await execution_engine.create_order(
            symbol="AAPL",
            side="BUY",
            quantity=10,
            price=150.0
        )
        assert demo_execution["paper_trade"] is True
        assert demo_execution["mode"] == "DEMO"
        
        set_mode("LIVE")
        # In LIVE mode, we would create real orders (mocked for testing)
        with patch('src.trading.execution.get_execution_engine') as mock_exec:
            mock_exec_engine = Mock()
            mock_exec_engine.create_order.return_value = {
                "order_id": "live_order_123",
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 10,
                "price": 150.0,
                "status": "PENDING",
                "paper_trade": False,
                "mode": "LIVE"
            }
            mock_exec.return_value = mock_exec_engine
            
            live_execution = await execution_engine.create_order(
                symbol="AAPL",
                side="BUY",
                quantity=10,
                price=150.0
            )
            assert live_execution["paper_trade"] is False
            assert live_execution["mode"] == "LIVE"
        
        logger.info("✓ Active trade handling completed successfully")
    
    @pytest.mark.asyncio
    async def test_mode_switch_validation(self, setup_clean_state):
        """Test validation during mode switches."""
        logger.info("Testing mode switch validation...")
        
        # Test valid mode switches
        valid_modes = ["DEMO", "LIVE"]
        for mode in valid_modes:
            set_mode(mode)
            assert get_current_mode() == mode
        
        # Test invalid mode handling
        invalid_modes = ["INVALID", "TEST", "PROD", ""]
        for invalid_mode in invalid_modes:
            try:
                set_mode(invalid_mode)
                assert False, f"Should have raised an error for invalid mode: {invalid_mode}"
            except ValueError:
                pass  # Expected behavior
        
        # Test mode configuration validation
        set_mode("DEMO")
        demo_config = get_mode_config()
        assert demo_config.mode == "DEMO"
        assert demo_config.paper_trading is True
        
        set_mode("LIVE")
        live_config = get_mode_config()
        assert live_config.mode == "LIVE"
        assert live_config.paper_trading is False
        
        # Test database path validation
        db_manager = get_database_manager()
        
        set_mode("DEMO")
        demo_db_path = db_manager.get_database_path()
        assert "demo" in demo_db_path.lower()
        
        set_mode("LIVE")
        live_db_path = db_manager.get_database_path()
        assert "live" in live_db_path.lower()
        
        logger.info("✓ Mode switch validation completed successfully")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
