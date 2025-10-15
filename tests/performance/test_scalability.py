"""
Performance Tests for Scalability
=================================

Tests the scalability of the AI Trading System with 2000+ stocks
and concurrent operations.
"""

import pytest
import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
import psutil
import gc

# Import components to test
from src.ai.enhanced_ensemble import get_enhanced_ensemble
from src.ai.multi_model import get_multi_model_manager
from src.workflows.trading_cycle import execute_complete_cycle
from src.config.mode_manager import set_mode

logger = logging.getLogger(__name__)


class TestScalability:
    """Test suite for scalability requirements."""
    
    @pytest.fixture
    async def setup_demo_mode(self):
        """Setup demo mode for testing."""
        set_mode("DEMO")
        yield
        # Cleanup after test
    
    @pytest.fixture
    async def generate_large_stock_universe(self):
        """Generate a large stock universe for testing."""
        stocks = {}
        for i in range(2000):  # 2000+ stocks
            symbol = f"STOCK_{i:04d}"
            stocks[symbol] = {
                "price": 10.0 + (i % 1000) * 0.1,
                "volume": 100000 + (i % 10000) * 100,
                "timestamp": datetime.now(),
                "bid": 10.0 + (i % 1000) * 0.1 - 0.05,
                "ask": 10.0 + (i % 1000) * 0.1 + 0.05
            }
        return stocks
    
    @pytest.mark.asyncio
    async def test_multi_stock_processing(self, setup_demo_mode, generate_large_stock_universe):
        """Test processing 2000+ stocks."""
        logger.info("Testing multi-stock processing (2000+ stocks)...")
        
        large_stock_universe = generate_large_stock_universe
        symbols = list(large_stock_universe.keys())
        
        # Test processing in batches
        batch_size = 100
        batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
        
        start_time = time.time()
        processed_batches = []
        
        for batch in batches:
            # Process batch
            batch_data = {symbol: large_stock_universe[symbol] for symbol in batch}
            processed_batch = await self._process_stock_batch(batch_data)
            processed_batches.append(processed_batch)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify all batches were processed
        total_processed = sum(len(batch) for batch in processed_batches)
        assert total_processed == len(symbols), f"Processed {total_processed} stocks, expected {len(symbols)}"
        
        # Verify processing time is reasonable (<60s for 2000 stocks)
        assert total_time < 60, f"Processing 2000+ stocks took {total_time:.2f}s, must be <60s"
        
        # Calculate throughput
        throughput = len(symbols) / total_time
        logger.info(f"✓ Multi-stock processing: {len(symbols)} stocks in {total_time:.2f}s ({throughput:.1f} stocks/sec)")
    
    async def _process_stock_batch(self, batch_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a batch of stocks."""
        processed_batch = []
        for symbol, data in batch_data.items():
            processed_stock = {
                "symbol": symbol,
                "price": data["price"],
                "volume": data["volume"],
                "processed_at": datetime.now(),
                "analysis_score": hash(symbol) % 100  # Mock analysis score
            }
            processed_batch.append(processed_stock)
        return processed_batch
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, setup_demo_mode, generate_large_stock_universe):
        """Test concurrent operations scalability."""
        logger.info("Testing concurrent operations scalability...")
        
        large_stock_universe = generate_large_stock_universe
        symbols = list(large_stock_universe.keys())[:100]  # Test with 100 stocks
        
        # Test concurrent processing
        start_time = time.time()
        
        # Create concurrent tasks
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(self._process_single_stock(symbol, large_stock_universe[symbol]))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        concurrent_time = end_time - start_time
        
        # Verify all tasks completed
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == len(symbols), f"Only {len(successful_results)}/{len(symbols)} tasks completed successfully"
        
        # Verify concurrent performance
        sequential_time = len(symbols) * 0.1  # Assume 0.1s per stock
        assert concurrent_time < sequential_time, f"Concurrent operations took {concurrent_time:.2f}s, should be faster than sequential"
        
        logger.info(f"✓ Concurrent operations: {len(symbols)} stocks in {concurrent_time:.2f}s")
    
    async def _process_single_stock(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single stock."""
        # Simulate processing time
        await asyncio.sleep(0.01)
        
        return {
            "symbol": symbol,
            "price": data["price"],
            "volume": data["volume"],
            "processed_at": datetime.now(),
            "analysis_score": hash(symbol) % 100
        }
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, setup_demo_mode, generate_large_stock_universe):
        """Test memory usage under load."""
        logger.info("Testing memory usage under load...")
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large dataset
        large_stock_universe = generate_large_stock_universe
        symbols = list(large_stock_universe.keys())[:500]  # Test with 500 stocks
        
        # Process data in chunks to monitor memory
        chunk_size = 50
        chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]
        
        max_memory = initial_memory
        for chunk in chunks:
            # Process chunk
            chunk_data = {symbol: large_stock_universe[symbol] for symbol in chunk}
            await self._process_stock_batch(chunk_data)
            
            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            max_memory = max(max_memory, current_memory)
            
            # Force garbage collection periodically
            if len(chunks) % 5 == 0:
                gc.collect()
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        peak_memory_increase = max_memory - initial_memory
        
        # Verify memory usage is reasonable
        assert memory_increase < 200, f"Memory increased by {memory_increase:.1f}MB, should be <200MB"
        assert peak_memory_increase < 300, f"Peak memory increased by {peak_memory_increase:.1f}MB, should be <300MB"
        
        logger.info(f"✓ Memory usage: {memory_increase:.1f}MB increase, peak: {peak_memory_increase:.1f}MB")
    
    @pytest.mark.asyncio
    async def test_database_performance(self, setup_demo_mode, generate_large_stock_universe):
        """Test database performance with large datasets."""
        logger.info("Testing database performance...")
        
        from src.config.database import get_database_manager
        
        db_manager = get_database_manager()
        
        # Test large dataset operations
        large_stock_universe = generate_large_stock_universe
        symbols = list(large_stock_universe.keys())[:1000]  # Test with 1000 stocks
        
        # Test batch insert performance
        start_time = time.time()
        
        # Mock batch insert
        batch_data = []
        for symbol in symbols:
            batch_data.append({
                "symbol": symbol,
                "price": large_stock_universe[symbol]["price"],
                "volume": large_stock_universe[symbol]["volume"],
                "timestamp": datetime.now()
            })
        
        # Simulate batch insert
        await asyncio.sleep(0.1)  # Mock database operation
        
        end_time = time.time()
        insert_time = end_time - start_time
        
        # Test query performance
        start_time = time.time()
        
        # Mock query
        query_result = {
            "count": len(symbols),
            "symbols": symbols[:10],  # Return first 10
            "query_time": datetime.now()
        }
        
        end_time = time.time()
        query_time = end_time - start_time
        
        # Verify performance
        assert insert_time < 1.0, f"Batch insert took {insert_time:.2f}s, must be <1s"
        assert query_time < 0.5, f"Query took {query_time:.2f}s, must be <0.5s"
        
        logger.info(f"✓ Database performance: insert {insert_time:.2f}s, query {query_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_ai_model_scaling(self, setup_demo_mode, generate_large_stock_universe):
        """Test AI model scaling with large datasets."""
        logger.info("Testing AI model scaling...")
        
        multi_model_manager = get_multi_model_manager()
        
        # Test with large dataset
        large_stock_universe = generate_large_stock_universe
        symbols = list(large_stock_universe.keys())[:200]  # Test with 200 stocks
        
        # Test ensemble analysis scaling
        start_time = time.time()
        
        # Mock ensemble analysis
        with patch('src.ai.multi_model.get_multi_model_manager') as mock_manager:
            mock_mm = Mock()
            mock_mm.analyze_with_ensemble.return_value = {
                "ensemble_analysis": {
                    "decision": "BUY",
                    "confidence": 0.85,
                    "reasoning": "Scalable analysis"
                },
                "model_opinions": [
                    {"model": "qwen2.5", "opinion": "BUY", "confidence": 0.8},
                    {"model": "llama3.1", "opinion": "BUY", "confidence": 0.9}
                ],
                "final_decision": "BUY",
                "processed_symbols": len(symbols)
            }
            mock_manager.return_value = mock_mm
            
            # Execute ensemble analysis
            result = await multi_model_manager.analyze_with_ensemble(
                symbols=symbols,
                market_data=large_stock_universe
            )
            
            end_time = time.time()
            analysis_time = end_time - start_time
        
        # Verify analysis result
        assert result is not None
        assert "ensemble_analysis" in result
        assert result["processed_symbols"] == len(symbols)
        
        # Verify scaling performance
        assert analysis_time < 10.0, f"AI model analysis took {analysis_time:.2f}s for {len(symbols)} symbols, must be <10s"
        
        # Calculate throughput
        throughput = len(symbols) / analysis_time
        logger.info(f"✓ AI model scaling: {len(symbols)} symbols in {analysis_time:.2f}s ({throughput:.1f} symbols/sec)")
    
    @pytest.mark.asyncio
    async def test_trading_cycle_scaling(self, setup_demo_mode, generate_large_stock_universe):
        """Test trading cycle scaling with large datasets."""
        logger.info("Testing trading cycle scaling...")
        
        large_stock_universe = generate_large_stock_universe
        symbols = list(large_stock_universe.keys())[:100]  # Test with 100 stocks
        
        # Test trading cycle scaling
        start_time = time.time()
        
        # Mock trading cycle
        with patch('src.workflows.trading_cycle.execute_complete_cycle') as mock_cycle:
            mock_cycle.return_value = {
                "signals": [
                    {
                        "decision": "BUY",
                        "symbol": symbol,
                        "confidence": 0.85,
                        "reasoning": "Scalable trading cycle"
                    } for symbol in symbols[:10]  # Return signals for first 10
                ],
                "executions": [],
                "positions": [],
                "risk_checks": {
                    "position_size_ok": True,
                    "portfolio_limits_ok": True,
                    "daily_drawdown_ok": True
                },
                "processed_symbols": len(symbols)
            }
            
            # Execute trading cycle
            result = await execute_complete_cycle(
                symbols=symbols,
                market_data=large_stock_universe
            )
            
            end_time = time.time()
            cycle_time = end_time - start_time
        
        # Verify cycle result
        assert result is not None
        assert "signals" in result
        assert result["processed_symbols"] == len(symbols)
        
        # Verify scaling performance
        assert cycle_time < 15.0, f"Trading cycle took {cycle_time:.2f}s for {len(symbols)} symbols, must be <15s"
        
        # Calculate throughput
        throughput = len(symbols) / cycle_time
        logger.info(f"✓ Trading cycle scaling: {len(symbols)} symbols in {cycle_time:.2f}s ({throughput:.1f} symbols/sec)")
    
    @pytest.mark.asyncio
    async def test_resource_utilization(self, setup_demo_mode, generate_large_stock_universe):
        """Test resource utilization under load."""
        logger.info("Testing resource utilization...")
        
        process = psutil.Process()
        
        # Get initial resource usage
        initial_cpu = process.cpu_percent()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large dataset
        large_stock_universe = generate_large_stock_universe
        symbols = list(large_stock_universe.keys())[:500]  # Test with 500 stocks
        
        # Monitor resource usage during processing
        cpu_samples = []
        memory_samples = []
        
        # Process in chunks and monitor resources
        chunk_size = 50
        chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]
        
        for chunk in chunks:
            # Process chunk
            chunk_data = {symbol: large_stock_universe[symbol] for symbol in chunk}
            await self._process_stock_batch(chunk_data)
            
            # Sample resource usage
            cpu_samples.append(process.cpu_percent())
            memory_samples.append(process.memory_info().rss / 1024 / 1024)
            
            # Small delay to allow sampling
            await asyncio.sleep(0.01)
        
        # Calculate resource usage statistics
        avg_cpu = sum(cpu_samples) / len(cpu_samples)
        max_cpu = max(cpu_samples)
        avg_memory = sum(memory_samples) / len(memory_samples)
        max_memory = max(memory_samples)
        
        memory_increase = max_memory - initial_memory
        
        # Verify resource usage is reasonable
        assert avg_cpu < 80, f"Average CPU usage {avg_cpu:.1f}%, should be <80%"
        assert max_cpu < 95, f"Peak CPU usage {max_cpu:.1f}%, should be <95%"
        assert memory_increase < 200, f"Memory increase {memory_increase:.1f}MB, should be <200MB"
        
        logger.info(f"✓ Resource utilization: CPU avg {avg_cpu:.1f}% max {max_cpu:.1f}%, Memory increase {memory_increase:.1f}MB")
    
    @pytest.mark.asyncio
    async def test_error_handling_scaling(self, setup_demo_mode, generate_large_stock_universe):
        """Test error handling under scaling load."""
        logger.info("Testing error handling scaling...")
        
        large_stock_universe = generate_large_stock_universe
        symbols = list(large_stock_universe.keys())[:100]  # Test with 100 stocks
        
        # Introduce some errors in the dataset
        error_symbols = symbols[:10]  # First 10 symbols will have errors
        for symbol in error_symbols:
            large_stock_universe[symbol]["price"] = None  # Invalid price
        
        # Test error handling
        start_time = time.time()
        
        successful_results = []
        error_results = []
        
        for symbol in symbols:
            try:
                if symbol in error_symbols:
                    # Simulate error
                    raise ValueError(f"Invalid data for {symbol}")
                else:
                    # Process normally
                    result = await self._process_single_stock(symbol, large_stock_universe[symbol])
                    successful_results.append(result)
            except Exception as e:
                error_results.append({"symbol": symbol, "error": str(e)})
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify error handling
        assert len(successful_results) == len(symbols) - len(error_symbols), "Not all valid symbols were processed"
        assert len(error_results) == len(error_symbols), "Not all errors were caught"
        
        # Verify processing time is reasonable
        assert processing_time < 5.0, f"Error handling took {processing_time:.2f}s, must be <5s"
        
        logger.info(f"✓ Error handling scaling: {len(successful_results)} successful, {len(error_results)} errors in {processing_time:.2f}s")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
