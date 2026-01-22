"""
Basic test script to verify the main orchestrator implementation structure.
Tests the core classes without full system initialization.
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all orchestrator components can be imported."""
    logger.info("Testing imports...")
    
    try:
        from continuous_data_collection.core.continuous_collector import ContinuousCollector
        from continuous_data_collection.core.lifecycle_manager import LifecycleManager, SystemLifecycleState
        from continuous_data_collection.core.completion_manager import CompletionManager, CompletionCriteria
        from continuous_data_collection.core.config import SystemConfig, ConfigLoader
        from continuous_data_collection.core.exceptions import (
            SystemInitializationError, CollectionError, LifecycleError,
            ConfigurationError, ReportingError, CompletionError
        )
        
        logger.info("✓ All imports successful")
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_config_creation():
    """Test configuration creation."""
    logger.info("Testing configuration creation...")
    
    try:
        from continuous_data_collection.core.config import SystemConfig
        
        # Create default config
        config = SystemConfig()
        assert config.environment == "development"
        assert config.collection.target_stocks_count == 1400
        assert config.collection.batch_size == 50
        
        # Create custom config
        config = SystemConfig(environment="testing")
        config.collection.target_stocks_count = 10
        assert config.collection.target_stocks_count == 10
        
        logger.info("✓ Configuration creation test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration creation test failed: {e}")
        return False


def test_completion_criteria():
    """Test completion criteria creation."""
    logger.info("Testing completion criteria...")
    
    try:
        from continuous_data_collection.core.completion_manager import CompletionCriteria
        
        # Create default criteria
        criteria = CompletionCriteria()
        assert criteria.min_completion_percentage == 95.0
        assert criteria.max_pending_stocks == 10
        assert criteria.require_high_priority_complete == True
        
        # Create custom criteria
        criteria = CompletionCriteria(
            min_completion_percentage=90.0,
            max_pending_stocks=5
        )
        assert criteria.min_completion_percentage == 90.0
        assert criteria.max_pending_stocks == 5
        
        logger.info("✓ Completion criteria test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Completion criteria test failed: {e}")
        return False


def test_lifecycle_states():
    """Test lifecycle state enumeration."""
    logger.info("Testing lifecycle states...")
    
    try:
        from continuous_data_collection.core.lifecycle_manager import SystemLifecycleState
        
        # Check all expected states exist
        expected_states = [
            "STOPPED", "STARTING", "RUNNING", "PAUSING", "PAUSED",
            "RESUMING", "STOPPING", "ERROR", "EMERGENCY_SHUTDOWN"
        ]
        
        for state_name in expected_states:
            state = getattr(SystemLifecycleState, state_name)
            assert state.value == state_name.lower()
        
        logger.info("✓ Lifecycle states test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Lifecycle states test failed: {e}")
        return False


def test_exception_hierarchy():
    """Test exception hierarchy."""
    logger.info("Testing exception hierarchy...")
    
    try:
        from continuous_data_collection.core.exceptions import (
            ContinuousCollectionError, SystemInitializationError,
            CollectionError, LifecycleError, ConfigurationError,
            ReportingError, CompletionError
        )
        
        # Test inheritance
        assert issubclass(SystemInitializationError, ContinuousCollectionError)
        assert issubclass(CollectionError, ContinuousCollectionError)
        assert issubclass(LifecycleError, ContinuousCollectionError)
        assert issubclass(ConfigurationError, ContinuousCollectionError)
        assert issubclass(ReportingError, ContinuousCollectionError)
        assert issubclass(CompletionError, ContinuousCollectionError)
        
        # Test exception creation
        error = SystemInitializationError("Test error")
        assert str(error) == "Test error"
        
        logger.info("✓ Exception hierarchy test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Exception hierarchy test failed: {e}")
        return False


def test_data_models():
    """Test data model creation."""
    logger.info("Testing data models...")
    
    try:
        from continuous_data_collection.core.models import (
            SystemState, ProgressStats, CollectionResult, DataSource, Priority
        )
        
        # Test SystemState
        state = SystemState()
        assert isinstance(state.completed_stocks, set)
        assert isinstance(state.failed_stocks, dict)
        assert isinstance(state.pending_stocks, list)
        
        # Test ProgressStats
        stats = ProgressStats(
            total_target=100,
            completed=50,
            failed=5,
            pending=45
        )
        assert stats.total_target == 100
        assert stats.completed == 50
        
        # Test CollectionResult
        result = CollectionResult(
            symbol="AAPL",
            success=True,
            source=DataSource.ALPHA_VANTAGE,
            data_quality_score=0.8,
            years_of_data=15.0,
            record_count=3000
        )
        assert result.symbol == "AAPL"
        assert result.success == True
        
        logger.info("✓ Data models test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Data models test failed: {e}")
        return False


async def test_async_interfaces():
    """Test async interface definitions."""
    logger.info("Testing async interfaces...")
    
    try:
        from continuous_data_collection.core.interfaces import (
            IContinuousCollector, IProgressTracker, IHealthMonitor
        )
        
        # Check that interfaces have expected methods
        assert hasattr(IContinuousCollector, 'start_collection')
        assert hasattr(IContinuousCollector, 'stop_collection')
        assert hasattr(IContinuousCollector, 'pause_collection')
        assert hasattr(IContinuousCollector, 'resume_collection')
        
        assert hasattr(IProgressTracker, 'update_progress')
        assert hasattr(IProgressTracker, 'get_current_stats')
        
        assert hasattr(IHealthMonitor, 'check_system_health')
        
        logger.info("✓ Async interfaces test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Async interfaces test failed: {e}")
        return False


async def main():
    """Run all basic tests."""
    logger.info("Starting basic orchestrator implementation tests...")
    
    tests = [
        test_imports,
        test_config_creation,
        test_completion_criteria,
        test_lifecycle_states,
        test_exception_hierarchy,
        test_data_models,
        test_async_interfaces
    ]
    
    results = []
    for test in tests:
        try:
            if asyncio.iscoroutinefunction(test):
                result = await test()
            else:
                result = test()
            results.append(result)
        except Exception as e:
            logger.error(f"Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    logger.info(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All basic orchestrator implementation tests passed!")
        return True
    else:
        logger.error("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)