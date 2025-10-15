"""
Integration Tests for Trading Cycle
===================================

Tests the complete trading cycle workflow from signal generation
to execution and position management.
"""

import pytest
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock

# Import trading cycle components
from src.workflows.trading_cycle import get_trading_cycle, execute_complete_cycle
from src.trading.execution import get_execution_engine
from src.trading.positions import get_position_manager
from src.trading.risk import get_risk_manager
from src.ai.enhanced_ensemble import get_enhanced_ensemble
from src.config.mode_manager import set_mode, get_current_mode

logger = logging.getLogger(__name__)


class TestTradingCycle:
    """Test suite for trading cycle integration."""
    
    @pytest.fixture
    async def setup_demo_mode(self):
        """Setup demo mode for testing."""
        set_mode("DEMO")
        yield
        # Cleanup after test
    
    @pytest.fixture
    async def setup_live_mode(self):
        """Setup live mode for testing."""
        set_mode("LIVE")
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
            },
            "MSFT": {
                "price": 300.0,
                "volume": 500000,
                "timestamp": datetime.now(),
                "bid": 299.95,
                "ask": 300.05
            }
        }
    
    @pytest.fixture
    async def mock_ai_signal(self):
        """Mock AI trading signal."""
        return {
            "decision": "BUY",
            "symbol": "AAPL",
            "confidence": 0.85,
            "reasoning": "Strong technical indicators and positive sentiment",
            "entry_price": 150.0,
            "stop_loss": 145.0,
            "take_profit": 160.0,
            "position_size": 0.1
        }
    
    @pytest.mark.asyncio
    async def test_buy_signal_to_execution(self, setup_demo_mode, mock_market_data, mock_ai_signal):
        """Test complete buy signal to execution flow."""
        logger.info("Testing buy signal to execution flow...")
        
        # Get trading cycle components
        trading_cycle = get_trading_cycle()
        execution_engine = get_execution_engine()
        position_manager = get_position_manager()
        
        # Mock AI ensemble to return buy signal
        with patch('src.ai.enhanced_ensemble.get_enhanced_ensemble') as mock_ensemble:
            mock_ai = Mock()
            mock_ai.analyze_for_entry.return_value = mock_ai_signal
            mock_ensemble.return_value = mock_ai
            
            # Execute trading cycle
            result = await execute_complete_cycle(
                symbols=["AAPL"],
                market_data=mock_market_data
            )
            
            # Verify results
            assert result is not None
            assert "signals" in result
            assert "executions" in result
            assert "positions" in result
            
            # Verify signal was generated
            signals = result["signals"]
            assert len(signals) > 0
            assert signals[0]["decision"] == "BUY"
            assert signals[0]["symbol"] == "AAPL"
            
            # Verify execution occurred
            executions = result["executions"]
            assert len(executions) > 0
            assert executions[0]["symbol"] == "AAPL"
            assert executions[0]["side"] == "BUY"
            
            logger.info("✓ Buy signal to execution flow completed successfully")
    
    @pytest.mark.asyncio
    async def test_position_tracking(self, setup_demo_mode, mock_market_data):
        """Test position tracking throughout trading cycle."""
        logger.info("Testing position tracking...")
        
        position_manager = get_position_manager()
        
        # Create a test position
        test_position = {
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10,
            "entry_price": 150.0,
            "entry_time": datetime.now(),
            "stop_loss": 145.0,
            "take_profit": 160.0,
            "mode": "DEMO"
        }
        
        # Track position
        await position_manager.track_position(test_position)
        
        # Verify position is tracked
        open_positions = await position_manager.get_open_positions()
        assert len(open_positions) > 0
        
        position = open_positions[0]
        assert position["symbol"] == "AAPL"
        assert position["quantity"] == 10
        assert position["entry_price"] == 150.0
        
        logger.info("✓ Position tracking completed successfully")
    
    @pytest.mark.asyncio
    async def test_hold_period_monitoring(self, setup_demo_mode, mock_market_data):
        """Test hold period monitoring and management."""
        logger.info("Testing hold period monitoring...")
        
        position_manager = get_position_manager()
        
        # Create a position with specific entry time
        entry_time = datetime.now() - timedelta(hours=2)  # 2 hours ago
        test_position = {
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10,
            "entry_price": 150.0,
            "entry_time": entry_time,
            "stop_loss": 145.0,
            "take_profit": 160.0,
            "mode": "DEMO"
        }
        
        # Track position
        await position_manager.track_position(test_position)
        
        # Get position summary
        portfolio_summary = await position_manager.get_portfolio_summary()
        
        # Verify hold period is tracked
        assert "positions" in portfolio_summary
        positions = portfolio_summary["positions"]
        assert len(positions) > 0
        
        position = positions[0]
        assert "hold_duration" in position
        assert position["hold_duration"] > 0  # Should be > 0 for 2-hour old position
        
        logger.info("✓ Hold period monitoring completed successfully")
    
    @pytest.mark.asyncio
    async def test_sell_signal_generation(self, setup_demo_mode, mock_market_data):
        """Test sell signal generation for open positions."""
        logger.info("Testing sell signal generation...")
        
        # Get components
        trading_cycle = get_trading_cycle()
        position_manager = get_position_manager()
        
        # Create an open position
        test_position = {
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10,
            "entry_price": 150.0,
            "entry_time": datetime.now() - timedelta(hours=1),
            "stop_loss": 145.0,
            "take_profit": 160.0,
            "mode": "DEMO"
        }
        
        await position_manager.track_position(test_position)
        
        # Mock AI ensemble to return sell signal
        mock_sell_signal = {
            "decision": "SELL",
            "symbol": "AAPL",
            "confidence": 0.90,
            "reasoning": "Take profit target reached",
            "exit_price": 160.0
        }
        
        with patch('src.ai.enhanced_ensemble.get_enhanced_ensemble') as mock_ensemble:
            mock_ai = Mock()
            mock_ai.analyze_for_exit.return_value = mock_sell_signal
            mock_ensemble.return_value = mock_ai
            
            # Execute trading cycle for exit
            result = await execute_complete_cycle(
                symbols=["AAPL"],
                market_data=mock_market_data
            )
            
            # Verify sell signal was generated
            assert result is not None
            signals = result.get("signals", [])
            
            # Should have at least one signal (could be buy or sell)
            assert len(signals) > 0
            
            # Check if any signal is a sell signal
            sell_signals = [s for s in signals if s.get("decision") == "SELL"]
            if sell_signals:
                assert sell_signals[0]["symbol"] == "AAPL"
                assert sell_signals[0]["confidence"] > 0.8
            
            logger.info("✓ Sell signal generation completed successfully")
    
    @pytest.mark.asyncio
    async def test_complete_cycle_demo_mode(self, setup_demo_mode, mock_market_data):
        """Test complete trading cycle in demo mode."""
        logger.info("Testing complete cycle in demo mode...")
        
        # Verify we're in demo mode
        assert get_current_mode() == "DEMO"
        
        # Mock AI signals
        mock_signals = [
            {
                "decision": "BUY",
                "symbol": "AAPL",
                "confidence": 0.85,
                "reasoning": "Strong buy signal",
                "entry_price": 150.0,
                "stop_loss": 145.0,
                "take_profit": 160.0,
                "position_size": 0.1
            }
        ]
        
        with patch('src.ai.enhanced_ensemble.get_enhanced_ensemble') as mock_ensemble:
            mock_ai = Mock()
            mock_ai.analyze_for_entry.return_value = mock_signals[0]
            mock_ensemble.return_value = mock_ai
            
            # Execute complete cycle
            result = await execute_complete_cycle(
                symbols=["AAPL"],
                market_data=mock_market_data
            )
            
            # Verify complete cycle results
            assert result is not None
            assert "signals" in result
            assert "executions" in result
            assert "positions" in result
            assert "risk_checks" in result
            
            # Verify demo mode specific behavior
            executions = result["executions"]
            if executions:
                assert executions[0]["mode"] == "DEMO"
                assert executions[0]["paper_trade"] is True
            
            logger.info("✓ Complete cycle in demo mode completed successfully")
    
    @pytest.mark.asyncio
    async def test_complete_cycle_live_mode(self, setup_live_mode, mock_market_data):
        """Test complete trading cycle in live mode."""
        logger.info("Testing complete cycle in live mode...")
        
        # Verify we're in live mode
        assert get_current_mode() == "LIVE"
        
        # Mock AI signals
        mock_signals = [
            {
                "decision": "BUY",
                "symbol": "AAPL",
                "confidence": 0.85,
                "reasoning": "Strong buy signal",
                "entry_price": 150.0,
                "stop_loss": 145.0,
                "take_profit": 160.0,
                "position_size": 0.1
            }
        ]
        
        with patch('src.ai.enhanced_ensemble.get_enhanced_ensemble') as mock_ensemble:
            mock_ai = Mock()
            mock_ai.analyze_for_entry.return_value = mock_signals[0]
            mock_ensemble.return_value = mock_ai
            
            # Mock execution engine to prevent real trades
            with patch('src.trading.execution.get_execution_engine') as mock_exec:
                mock_exec_engine = Mock()
                mock_exec_engine.create_order.return_value = {
                    "order_id": "test_order_123",
                    "symbol": "AAPL",
                    "side": "BUY",
                    "quantity": 10,
                    "price": 150.0,
                    "status": "FILLED",
                    "mode": "LIVE"
                }
                mock_exec.return_value = mock_exec_engine
                
                # Execute complete cycle
                result = await execute_complete_cycle(
                    symbols=["AAPL"],
                    market_data=mock_market_data
                )
                
                # Verify complete cycle results
                assert result is not None
                assert "signals" in result
                assert "executions" in result
                assert "positions" in result
                assert "risk_checks" in result
                
                # Verify live mode specific behavior
                executions = result["executions"]
                if executions:
                    assert executions[0]["mode"] == "LIVE"
                    assert executions[0]["paper_trade"] is False
                
                logger.info("✓ Complete cycle in live mode completed successfully")
    
    @pytest.mark.asyncio
    async def test_risk_management_integration(self, setup_demo_mode, mock_market_data):
        """Test risk management integration in trading cycle."""
        logger.info("Testing risk management integration...")
        
        risk_manager = get_risk_manager()
        
        # Test position sizing
        portfolio_value = 10000.0
        symbol = "AAPL"
        price = 150.0
        confidence = 0.85
        
        position_size = await risk_manager.calculate_position_size(
            portfolio_value=portfolio_value,
            symbol=symbol,
            price=price,
            confidence=confidence
        )
        
        # Verify position size is calculated
        assert position_size > 0
        assert position_size <= portfolio_value * 0.25  # Max 25% per position
        
        # Test portfolio limits
        portfolio_limits = await risk_manager.check_portfolio_limits(
            current_positions=[],
            new_position={"symbol": symbol, "value": position_size}
        )
        
        # Verify portfolio limits are checked
        assert portfolio_limits is not None
        assert "within_limits" in portfolio_limits
        
        # Test daily drawdown
        daily_pnl = -500.0  # $500 loss
        drawdown_check = await risk_manager.check_daily_drawdown(daily_pnl)
        
        # Verify drawdown check
        assert drawdown_check is not None
        assert "within_limits" in drawdown_check
        
        logger.info("✓ Risk management integration completed successfully")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
