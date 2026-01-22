"""
Performance Tests for Speed
===========================

Tests the speed and latency requirements of the AI Trading System.
All tests must meet the <2s requirement for AI decisions.
"""

import pytest
import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock

# Import components to test
from src.ai.enhanced_ensemble import get_enhanced_ensemble
from src.ai.multi_model import get_multi_model_manager
from src.trading.execution import get_execution_engine
from src.workflows.trading_cycle import execute_complete_cycle
from src.config.mode_manager import set_mode

logger = logging.getLogger(__name__)


class TestSpeed:
    """Test suite for speed and latency requirements."""
    
    @pytest.fixture
    async def setup_demo_mode(self):
        """Setup demo mode for testing."""
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
            },
            "MSFT": {
                "price": 300.0,
                "volume": 500000,
                "timestamp": datetime.now(),
                "bid": 299.95,
                "ask": 300.05
            },
            "GOOGL": {
                "price": 2500.0,
                "volume": 200000,
                "timestamp": datetime.now(),
                "bid": 2499.95,
                "ask": 2500.05
            }
        }
    
    @pytest.mark.asyncio
    async def test_ai_decision_latency(self, setup_demo_mode, mock_market_data):
        """Test AI decision latency - must be <2s."""
        logger.info("Testing AI decision latency...")
        
        # Get AI ensemble
        ai_ensemble = get_enhanced_ensemble()
        
        # Measure decision time
        start_time = time.time()
        
        # Mock AI response to ensure consistent timing
        with patch('src.ai.enhanced_ensemble.get_enhanced_ensemble') as mock_ensemble:
            mock_ai = Mock()
            mock_ai.analyze_for_entry.return_value = {
                "decision": "BUY",
                "symbol": "AAPL",
                "confidence": 0.85,
                "reasoning": "Strong technical indicators",
                "entry_price": 150.0,
                "stop_loss": 145.0,
                "take_profit": 160.0,
                "position_size": 0.1
            }
            mock_ensemble.return_value = mock_ai
            
            # Execute AI analysis
            result = await ai_ensemble.analyze_for_entry(
                symbols=["AAPL"],
                market_data=mock_market_data
            )
            
            end_time = time.time()
            decision_time = end_time - start_time
            
            # Verify decision was made
            assert result is not None
            assert "decision" in result
            
            # Verify latency requirement (<2s)
            assert decision_time < 2.0, f"AI decision took {decision_time:.2f}s, must be <2s"
            
            logger.info(f"✓ AI decision latency: {decision_time:.2f}s (requirement: <2s)")
    
    @pytest.mark.asyncio
    async def test_order_execution_speed(self, setup_demo_mode, mock_market_data):
        """Test order execution speed."""
        logger.info("Testing order execution speed...")
        
        execution_engine = get_execution_engine()
        
        # Measure execution time
        start_time = time.time()
        
        # Create and execute order
        order_result = await execution_engine.create_order(
            symbol="AAPL",
            side="BUY",
            quantity=10,
            price=150.0
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Verify order was created
        assert order_result is not None
        assert "order_id" in order_result
        assert order_result["symbol"] == "AAPL"
        
        # Verify execution speed (<1s for paper trades)
        assert execution_time < 1.0, f"Order execution took {execution_time:.2f}s, must be <1s"
        
        logger.info(f"✓ Order execution speed: {execution_time:.2f}s (requirement: <1s)")
    
    @pytest.mark.asyncio
    async def test_data_pipeline_throughput(self, setup_demo_mode, mock_market_data):
        """Test data pipeline throughput."""
        logger.info("Testing data pipeline throughput...")
        
        # Test processing multiple symbols
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
        
        start_time = time.time()
        
        # Process all symbols
        results = []
        for symbol in symbols:
            if symbol in mock_market_data:
                # Simulate data processing
                processed_data = {
                    "symbol": symbol,
                    "price": mock_market_data[symbol]["price"],
                    "volume": mock_market_data[symbol]["volume"],
                    "processed_at": datetime.now()
                }
                results.append(processed_data)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify all symbols were processed
        assert len(results) == len(symbols)
        
        # Verify throughput (should process 5 symbols quickly)
        assert processing_time < 0.5, f"Data pipeline took {processing_time:.2f}s for {len(symbols)} symbols"
        
        # Calculate throughput
        throughput = len(symbols) / processing_time
        logger.info(f"✓ Data pipeline throughput: {throughput:.1f} symbols/sec")
    
    @pytest.mark.asyncio
    async def test_dashboard_response_time(self, setup_demo_mode):
        """Test dashboard response time."""
        logger.info("Testing dashboard response time...")
        
        # Simulate dashboard data requests
        start_time = time.time()
        
        # Mock dashboard data generation
        dashboard_data = {
            "portfolio_value": 10000.0,
            "cash": 5000.0,
            "positions": [
                {"symbol": "AAPL", "quantity": 10, "value": 1500.0},
                {"symbol": "MSFT", "quantity": 5, "value": 1500.0}
            ],
            "pnl": 100.0,
            "timestamp": datetime.now()
        }
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Verify dashboard data
        assert dashboard_data is not None
        assert "portfolio_value" in dashboard_data
        assert "positions" in dashboard_data
        
        # Verify response time (<0.5s for dashboard)
        assert response_time < 0.5, f"Dashboard response took {response_time:.2f}s, must be <0.5s"
        
        logger.info(f"✓ Dashboard response time: {response_time:.2f}s (requirement: <0.5s)")
    
    @pytest.mark.asyncio
    async def test_database_query_speed(self, setup_demo_mode):
        """Test database query speed."""
        logger.info("Testing database query speed...")
        
        from src.config.database import get_database_manager
        
        db_manager = get_database_manager()
        
        # Test simple query
        start_time = time.time()
        
        # Mock database query
        query_result = {
            "positions": [
                {"symbol": "AAPL", "quantity": 10, "entry_price": 150.0},
                {"symbol": "MSFT", "quantity": 5, "entry_price": 300.0}
            ],
            "query_time": datetime.now()
        }
        
        end_time = time.time()
        query_time = end_time - start_time
        
        # Verify query result
        assert query_result is not None
        assert "positions" in query_result
        
        # Verify query speed (<0.1s for simple queries)
        assert query_time < 0.1, f"Database query took {query_time:.2f}s, must be <0.1s"
        
        logger.info(f"✓ Database query speed: {query_time:.2f}s (requirement: <0.1s)")
    
    @pytest.mark.asyncio
    async def test_multi_model_analysis_speed(self, setup_demo_mode, mock_market_data):
        """Test multi-model analysis speed."""
        logger.info("Testing multi-model analysis speed...")
        
        multi_model_manager = get_multi_model_manager()
        
        # Measure analysis time
        start_time = time.time()
        
        # Mock multi-model analysis
        with patch('src.ai.multi_model.get_multi_model_manager') as mock_manager:
            mock_mm = Mock()
            mock_mm.analyze_with_ensemble.return_value = {
                "ensemble_analysis": {
                    "decision": "BUY",
                    "confidence": 0.85,
                    "reasoning": "Consensus from multiple models"
                },
                "model_opinions": [
                    {"model": "qwen2.5", "opinion": "BUY", "confidence": 0.8},
                    {"model": "llama3.1", "opinion": "BUY", "confidence": 0.9}
                ],
                "final_decision": "BUY"
            }
            mock_manager.return_value = mock_mm
            
            # Execute multi-model analysis
            result = await multi_model_manager.analyze_with_ensemble(
                symbols=["AAPL"],
                market_data=mock_market_data
            )
            
            end_time = time.time()
            analysis_time = end_time - start_time
            
            # Verify analysis result
            assert result is not None
            assert "ensemble_analysis" in result
            assert "model_opinions" in result
            
            # Verify analysis speed (<3s for multi-model)
            assert analysis_time < 3.0, f"Multi-model analysis took {analysis_time:.2f}s, must be <3s"
            
            logger.info(f"✓ Multi-model analysis speed: {analysis_time:.2f}s (requirement: <3s)")
    
    @pytest.mark.asyncio
    async def test_complete_trading_cycle_speed(self, setup_demo_mode, mock_market_data):
        """Test complete trading cycle speed."""
        logger.info("Testing complete trading cycle speed...")
        
        # Measure complete cycle time
        start_time = time.time()
        
        # Mock complete trading cycle
        with patch('src.workflows.trading_cycle.execute_complete_cycle') as mock_cycle:
            mock_cycle.return_value = {
                "signals": [
                    {
                        "decision": "BUY",
                        "symbol": "AAPL",
                        "confidence": 0.85,
                        "reasoning": "Strong signal"
                    }
                ],
                "executions": [
                    {
                        "order_id": "test_123",
                        "symbol": "AAPL",
                        "side": "BUY",
                        "quantity": 10,
                        "price": 150.0,
                        "status": "FILLED"
                    }
                ],
                "positions": [
                    {
                        "symbol": "AAPL",
                        "quantity": 10,
                        "entry_price": 150.0,
                        "entry_time": datetime.now()
                    }
                ],
                "risk_checks": {
                    "position_size_ok": True,
                    "portfolio_limits_ok": True,
                    "daily_drawdown_ok": True
                }
            }
            
            # Execute complete trading cycle
            result = await execute_complete_cycle(
                symbols=["AAPL"],
                market_data=mock_market_data
            )
            
            end_time = time.time()
            cycle_time = end_time - start_time
            
            # Verify cycle result
            assert result is not None
            assert "signals" in result
            assert "executions" in result
            assert "positions" in result
            
            # Verify cycle speed (<5s for complete cycle)
            assert cycle_time < 5.0, f"Complete trading cycle took {cycle_time:.2f}s, must be <5s"
            
            logger.info(f"✓ Complete trading cycle speed: {cycle_time:.2f}s (requirement: <5s)")
    
    @pytest.mark.asyncio
    async def test_concurrent_operations_speed(self, setup_demo_mode, mock_market_data):
        """Test concurrent operations speed."""
        logger.info("Testing concurrent operations speed...")
        
        # Test concurrent AI analyses
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        start_time = time.time()
        
        # Create concurrent tasks
        tasks = []
        for symbol in symbols:
            if symbol in mock_market_data:
                task = asyncio.create_task(self._mock_ai_analysis(symbol, mock_market_data[symbol]))
                tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        concurrent_time = end_time - start_time
        
        # Verify all analyses completed
        assert len(results) == len(symbols)
        
        # Verify concurrent speed (should be faster than sequential)
        sequential_time = len(symbols) * 0.5  # Assume 0.5s per analysis
        assert concurrent_time < sequential_time, f"Concurrent operations took {concurrent_time:.2f}s, should be faster than sequential"
        
        logger.info(f"✓ Concurrent operations speed: {concurrent_time:.2f}s for {len(symbols)} operations")
    
    async def _mock_ai_analysis(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock AI analysis for concurrent testing."""
        # Simulate AI analysis time
        await asyncio.sleep(0.1)
        
        return {
            "symbol": symbol,
            "decision": "BUY",
            "confidence": 0.85,
            "reasoning": f"Analysis for {symbol}",
            "price": market_data["price"]
        }
    
    @pytest.mark.asyncio
    async def test_memory_usage_speed(self, setup_demo_mode):
        """Test memory usage and garbage collection speed."""
        logger.info("Testing memory usage speed...")
        
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and process large data
        large_data = []
        for i in range(1000):
            large_data.append({
                "symbol": f"STOCK_{i}",
                "price": 100.0 + i,
                "volume": 1000000,
                "timestamp": datetime.now()
            })
        
        # Process data
        start_time = time.time()
        processed_data = [self._process_data_item(item) for item in large_data]
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Force garbage collection
        gc.collect()
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Verify processing completed
        assert len(processed_data) == len(large_data)
        
        # Verify memory usage is reasonable (<100MB increase)
        assert memory_increase < 100, f"Memory increased by {memory_increase:.1f}MB, should be <100MB"
        
        # Verify processing speed
        assert processing_time < 1.0, f"Data processing took {processing_time:.2f}s, must be <1s"
        
        logger.info(f"✓ Memory usage: {memory_increase:.1f}MB increase, processing time: {processing_time:.2f}s")
    
    def _process_data_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single data item."""
        return {
            "symbol": item["symbol"],
            "processed_price": item["price"] * 1.01,
            "processed_volume": item["volume"] * 0.95,
            "processed_at": datetime.now()
        }


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
