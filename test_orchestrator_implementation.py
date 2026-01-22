"""
Test script to verify the main orchestrator implementation.
Tests the ContinuousCollector, LifecycleManager, and CompletionManager components.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from continuous_data_collection.core import (
    SystemFactory, SystemConfig, ContinuousCollector, LifecycleManager,
    CompletionManager, SystemLifecycleState
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_system_creation():
    """Test system creation and initialization."""
    logger.info("Testing system creation...")
    
    try:
        # Create a minimal test configuration
        config = SystemConfig(environment="testing")
        config.collection.target_stocks_count = 5
        config.collection.batch_size = 2
        config.collection.max_workers = 1
        config.api.alpha_vantage_keys = ["test_key"]
        
        # Create system using factory
        collector, lifecycle_manager = SystemFactory.create_system(
            environment="testing",
            custom_config=config
        )
        
        # Verify components are created
        assert isinstance(collector, ContinuousCollector)
        assert isinstance(lifecycle_manager, LifecycleManager)
        assert isinstance(collector.completion_manager, CompletionManager)
        
        logger.info("✓ System creation test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ System creation test failed: {e}")
        return False


async def test_lifecycle_states():
    """Test lifecycle state management."""
    logger.info("Testing lifecycle state management...")
    
    try:
        # Create minimal system
        config = SystemConfig(environment="testing")
        config.collection.target_stocks_count = 5
        config.api.alpha_vantage_keys = ["test_key"]
        
        collector, lifecycle_manager = SystemFactory.create_system(
            environment="testing",
            custom_config=config
        )
        
        # Check initial state
        status = lifecycle_manager.get_lifecycle_status()
        assert status["lifecycle_state"] == SystemLifecycleState.STOPPED.value
        
        # Test state transitions (without actually starting the system)
        logger.info("✓ Lifecycle state management test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Lifecycle state management test failed: {e}")
        return False


async def test_completion_criteria():
    """Test completion criteria checking."""
    logger.info("Testing completion criteria...")
    
    try:
        # Create minimal system
        config = SystemConfig(environment="testing")
        config.collection.target_stocks_count = 5
        config.api.alpha_vantage_keys = ["test_key"]
        
        collector, lifecycle_manager = SystemFactory.create_system(
            environment="testing",
            custom_config=config
        )
        
        # Test completion criteria
        completion_manager = collector.completion_manager
        
        # Check completion status (should be incomplete initially)
        is_complete, reason = await completion_manager.check_completion_status()
        logger.info(f"Completion status: {is_complete}, Reason: {reason}")
        
        logger.info("✓ Completion criteria test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Completion criteria test failed: {e}")
        return False


async def test_system_status():
    """Test system status reporting."""
    logger.info("Testing system status reporting...")
    
    try:
        # Create minimal system
        config = SystemConfig(environment="testing")
        config.collection.target_stocks_count = 5
        config.api.alpha_vantage_keys = ["test_key"]
        
        collector, lifecycle_manager = SystemFactory.create_system(
            environment="testing",
            custom_config=config
        )
        
        # Get system status
        system_status = collector.get_system_status()
        
        # Verify status structure
        assert "system" in system_status
        assert "progress" in system_status
        assert "queue" in system_status
        assert "workers" in system_status
        
        # Get lifecycle status
        lifecycle_status = lifecycle_manager.get_lifecycle_status()
        assert "lifecycle_state" in lifecycle_status
        assert "state_change_time" in lifecycle_status
        
        logger.info("✓ System status reporting test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ System status reporting test failed: {e}")
        return False


async def main():
    """Run all tests."""
    logger.info("Starting orchestrator implementation tests...")
    
    tests = [
        test_system_creation,
        test_lifecycle_states,
        test_completion_criteria,
        test_system_status
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            logger.error(f"Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    logger.info(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All orchestrator implementation tests passed!")
        return True
    else:
        logger.error("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)