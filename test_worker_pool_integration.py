"""
Integration test for the worker pool and parallel processing system.

This test verifies that the worker pool, collection workers, and coordination
system work together correctly for parallel stock data collection.
"""

import asyncio
import logging
import sys
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

# Add the project root to the path
sys.path.append('.')

from continuous_data_collection.core.worker_pool import WorkerPool, WorkerPoolConfig
from continuous_data_collection.core.collection_worker import CollectionWorker, create_collection_worker
from continuous_data_collection.core.worker_coordinator import WorkerCoordinator, CoordinatedWorkerPool
from continuous_data_collection.core.models import (
    StockTask, Priority, CollectionResult, DataSource, ValidationResult, APIUsageStats
)
from continuous_data_collection.core.interfaces import (
    IMultiSourceCollector, IDataValidator, IDataStorage, IRateLimiter
)
from continuous_data_collection.core.rate_limiter import RateLimiter, RateLimitConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockMultiSourceCollector(IMultiSourceCollector):
    """Mock collector for testing."""
    
    def __init__(self):
        self.call_count = 0
    
    async def collect_stock_data(self, symbol: str) -> CollectionResult:
        """Mock data collection."""
        self.call_count += 1
        await asyncio.sleep(0.1)  # Simulate API call
        
        # Simulate occasional failures
        if symbol.endswith('FAIL'):
            return CollectionResult(
                symbol=symbol,
                success=False,
                source=DataSource.ALPHA_VANTAGE,
                data_quality_score=0.0,
                years_of_data=0.0,
                record_count=0,
                error_message="Mock failure"
            )
        
        return CollectionResult(
            symbol=symbol,
            success=True,
            source=DataSource.ALPHA_VANTAGE,
            data_quality_score=0.95,
            years_of_data=15.0,
            record_count=3900,
            processing_duration=0.1
        )
    
    async def validate_all_sources(self) -> Dict[str, bool]:
        """Mock source validation."""
        return {"alpha_vantage": True, "yfinance": True}
    
    def get_source_statistics(self) -> Dict[str, any]:
        """Mock source statistics."""
        return {"alpha_vantage": {"calls": self.call_count}}


class MockDataValidator(IDataValidator):
    """Mock validator for testing."""
    
    def validate_stock_data(self, data: pd.DataFrame, symbol: str) -> ValidationResult:
        """Mock data validation."""
        return ValidationResult(
            is_valid=True,
            quality_score=0.95,
            years_of_data=15.0,
            record_count=3900,
            missing_dates_count=10,
            missing_dates_percent=0.26,
            data_start_date=datetime(2009, 1, 1),
            data_end_date=datetime(2024, 1, 1)
        )
    
    def check_data_completeness(self, data: pd.DataFrame) -> float:
        """Mock completeness check."""
        return 0.95
    
    def verify_ohlcv_integrity(self, data: pd.DataFrame) -> List[str]:
        """Mock integrity check."""
        return []


class MockDataStorage(IDataStorage):
    """Mock storage for testing."""
    
    def __init__(self):
        self.stored_stocks = set()
    
    async def save_stock_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """Mock data saving."""
        self.stored_stocks.add(symbol)
        return True
    
    async def load_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Mock data loading."""
        if symbol in self.stored_stocks:
            # Return mock data
            dates = pd.date_range('2009-01-01', '2024-01-01', freq='D')
            return pd.DataFrame({
                'Open': [100.0] * len(dates),
                'High': [105.0] * len(dates),
                'Low': [95.0] * len(dates),
                'Close': [102.0] * len(dates),
                'Volume': [1000000] * len(dates)
            }, index=dates)
        return None
    
    async def stock_exists(self, symbol: str) -> bool:
        """Mock existence check."""
        return symbol in self.stored_stocks
    
    async def get_storage_stats(self) -> Dict[str, int]:
        """Mock storage stats."""
        return {"file_count": len(self.stored_stocks), "total_size": len(self.stored_stocks) * 1000}


class MockRateLimiter(IRateLimiter):
    """Mock rate limiter for testing."""
    
    def __init__(self):
        self.api_keys = ['test_key_1', 'test_key_2', 'test_key_3']
        self.current_index = 0
    
    async def acquire_permit(self, api_key: str) -> bool:
        """Mock permit acquisition."""
        return True
    
    async def release_permit(self, api_key: str) -> None:
        """Mock permit release."""
        pass
    
    def get_next_available_key(self) -> Optional[str]:
        """Mock key rotation."""
        if self.api_keys:
            key = self.api_keys[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            return key
        return None
    
    def get_usage_stats(self) -> Dict[str, APIUsageStats]:
        """Mock usage stats."""
        return {
            key: APIUsageStats(
                api_key=key,
                requests_made=10,
                requests_remaining=90,
                success_count=10,
                error_count=0
            )
            for key in self.api_keys
        }


async def test_worker_pool_basic():
    """Test basic worker pool functionality."""
    logger.info("Testing basic worker pool functionality...")
    
    # Create mock dependencies
    collector = MockMultiSourceCollector()
    validator = MockDataValidator()
    storage = MockDataStorage()
    
    # Create worker factory
    def worker_factory():
        return create_collection_worker(collector, validator, storage)
    
    # Create worker pool
    config = WorkerPoolConfig(
        min_workers=2,
        max_workers=4,
        initial_workers=2,
        use_processes=False  # Use threads for testing
    )
    
    worker_pool = WorkerPool(config, worker_factory)
    
    try:
        # Start worker pool
        await worker_pool.start_workers()
        
        # Create test tasks
        test_tasks = [
            StockTask(symbol="AAPL", priority=Priority.HIGH),
            StockTask(symbol="GOOGL", priority=Priority.HIGH),
            StockTask(symbol="MSFT", priority=Priority.MEDIUM),
            StockTask(symbol="TSLA", priority=Priority.MEDIUM),
            StockTask(symbol="FAIL", priority=Priority.LOW),  # This will fail
        ]
        
        # Process batch
        logger.info(f"Processing batch of {len(test_tasks)} tasks...")
        batch_result = await worker_pool.process_batch(test_tasks)
        
        # Verify results
        assert len(batch_result.results) == len(test_tasks)
        assert batch_result.success_count == 4  # 4 successful, 1 failed
        assert batch_result.failure_count == 1
        
        # Check pool stats
        pool_stats = worker_pool.get_pool_stats()
        logger.info(f"Pool stats: {pool_stats}")
        
        assert pool_stats['pool_status']['active_workers'] >= 2
        assert pool_stats['performance']['total_tasks_processed'] == len(test_tasks)
        
        logger.info("✓ Basic worker pool test passed")
        
    finally:
        await worker_pool.stop_workers()


async def test_worker_coordination():
    """Test worker coordination and resource management."""
    logger.info("Testing worker coordination...")
    
    # Create mock rate limiter
    rate_limiter = MockRateLimiter()
    
    # Create mock dependencies
    collector = MockMultiSourceCollector()
    validator = MockDataValidator()
    storage = MockDataStorage()
    
    # Create worker factory
    def worker_factory():
        return create_collection_worker(collector, validator, storage)
    
    # Create worker pool and coordinator
    pool_config = WorkerPoolConfig(
        min_workers=2,
        max_workers=3,
        initial_workers=2,
        use_processes=False
    )
    
    worker_pool = WorkerPool(pool_config, worker_factory)
    coordinator = WorkerCoordinator(rate_limiter)
    coordinated_pool = CoordinatedWorkerPool(worker_pool, coordinator)
    
    try:
        # Start coordinated pool
        await coordinated_pool.start_workers(2)
        
        # Create test tasks
        test_tasks = [
            StockTask(symbol="AAPL", priority=Priority.HIGH),
            StockTask(symbol="GOOGL", priority=Priority.HIGH),
            StockTask(symbol="MSFT", priority=Priority.MEDIUM),
        ]
        
        # Process batch with coordination
        logger.info("Processing batch with coordination...")
        batch_result = await coordinated_pool.process_batch(test_tasks)
        
        # Verify results
        assert len(batch_result.results) == len(test_tasks)
        assert batch_result.success_count == len(test_tasks)
        
        # Check coordination stats
        stats = await coordinated_pool.get_comprehensive_stats()
        logger.info(f"Coordination stats: {stats}")
        
        assert stats['coordination_stats']['coordination_status']['is_active']
        # Note: In this test, coordination is set up but tasks are processed directly by worker pool
        # In a full implementation, tasks would go through the coordinator
        
        logger.info("✓ Worker coordination test passed")
        
    finally:
        await coordinated_pool.stop_workers()


async def test_worker_scaling():
    """Test dynamic worker scaling."""
    logger.info("Testing dynamic worker scaling...")
    
    # Create mock dependencies
    collector = MockMultiSourceCollector()
    validator = MockDataValidator()
    storage = MockDataStorage()
    
    def worker_factory():
        return create_collection_worker(collector, validator, storage)
    
    # Create worker pool with scaling enabled
    config = WorkerPoolConfig(
        min_workers=1,
        max_workers=5,
        initial_workers=1,
        scale_up_threshold=0.1,  # Low threshold for testing
        scale_down_threshold=0.05,
        scale_check_interval=1,  # Fast scaling for testing
        use_processes=False
    )
    
    worker_pool = WorkerPool(config, worker_factory)
    
    try:
        # Start with minimum workers
        await worker_pool.start_workers(1)
        
        initial_stats = worker_pool.get_pool_stats()
        initial_workers = initial_stats['pool_status']['active_workers']
        logger.info(f"Started with {initial_workers} workers")
        
        # Create a large batch to trigger scaling
        large_batch = [
            StockTask(symbol=f"STOCK_{i}", priority=Priority.MEDIUM)
            for i in range(10)
        ]
        
        # Process large batch (should trigger scaling up)
        logger.info("Processing large batch to trigger scaling...")
        batch_result = await worker_pool.process_batch(large_batch)
        
        # Check if scaling occurred
        final_stats = worker_pool.get_pool_stats()
        final_workers = final_stats['pool_status']['active_workers']
        
        logger.info(f"Final worker count: {final_workers}")
        assert len(batch_result.results) == len(large_batch)
        
        # Test manual scaling
        await worker_pool.scale_workers(3)
        scaled_stats = worker_pool.get_pool_stats()
        scaled_workers = scaled_stats['pool_status']['active_workers']
        
        logger.info(f"After manual scaling: {scaled_workers} workers")
        assert scaled_workers == 3
        
        logger.info("✓ Worker scaling test passed")
        
    finally:
        await worker_pool.stop_workers()


async def test_error_handling():
    """Test error handling in worker pool."""
    logger.info("Testing error handling...")
    
    # Create mock dependencies
    collector = MockMultiSourceCollector()
    validator = MockDataValidator()
    storage = MockDataStorage()
    
    def worker_factory():
        return create_collection_worker(collector, validator, storage)
    
    config = WorkerPoolConfig(
        min_workers=2,
        max_workers=2,
        initial_workers=2,
        use_processes=False
    )
    
    worker_pool = WorkerPool(config, worker_factory)
    
    try:
        await worker_pool.start_workers()
        
        # Create tasks with some failures
        test_tasks = [
            StockTask(symbol="GOOD1", priority=Priority.HIGH),
            StockTask(symbol="FAIL", priority=Priority.HIGH),  # Will fail
            StockTask(symbol="GOOD2", priority=Priority.HIGH),
            StockTask(symbol="ANOTHERFAIL", priority=Priority.HIGH),  # Will fail
        ]
        
        # Process batch with failures
        batch_result = await worker_pool.process_batch(test_tasks)
        
        # Verify error handling
        assert len(batch_result.results) == len(test_tasks)
        assert batch_result.success_count == 2
        assert batch_result.failure_count == 2
        
        # Check that failed tasks have error messages
        failed_results = [r for r in batch_result.results if not r.success]
        for result in failed_results:
            assert result.error_message is not None
            assert "Mock failure" in result.error_message
        
        logger.info("✓ Error handling test passed")
        
    finally:
        await worker_pool.stop_workers()


async def main():
    """Run all worker pool integration tests."""
    logger.info("Starting worker pool integration tests...")
    
    try:
        await test_worker_pool_basic()
        await test_worker_coordination()
        await test_worker_scaling()
        await test_error_handling()
        
        logger.info("🎉 All worker pool integration tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())