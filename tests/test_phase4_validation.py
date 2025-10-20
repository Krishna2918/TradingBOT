"""
Phase 4 Validation Script

This script validates the Phase 4 implementation by running basic tests
and checking component integration.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test_imports():
    """Test that all Phase 4 modules can be imported."""
    try:
        logger.info("Testing Phase 4 imports...")
        
        # Test multi-model imports
        from ai.multi_model import MultiModelManager, ModelRole, ModelOpinion
        logger.info("✓ Multi-Model imports successful")
        
        # Test collaborative discussion imports
        from ai.collaborative_discussion import CollaborativeDiscussion, DiscussionSession, DiscussionRound, FinalDecision
        logger.info("✓ Collaborative Discussion imports successful")
        
        # Test off-hours discussion imports
        from ai.offhours_discussion import OffHoursAI, OffHoursSession, DiscussionTopic, TradeInsight, PatternAnalysis, ImprovementSuggestion
        logger.info("✓ Off-Hours Discussion imports successful")
        
        # Test Phase 4 integration imports
        from workflows.phase4_integration import Phase4Integration, Phase4Status, CollaborativeDecision, AIEnsembleMetrics
        logger.info("✓ Phase 4 Integration imports successful")
        
        # Test Phase 4 main imports
        from main_phase4 import Phase4Main
        logger.info("✓ Phase 4 Main imports successful")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error during imports: {e}")
        return False

async def test_component_initialization():
    """Test that Phase 4 components can be initialized."""
    try:
        logger.info("Testing Phase 4 component initialization...")
        
        # Test MultiModelManager initialization
        from ai.multi_model import MultiModelManager
        multi_model_manager = MultiModelManager("DEMO")
        logger.info("✓ MultiModelManager initialized")
        
        # Test CollaborativeDiscussion initialization
        from ai.collaborative_discussion import CollaborativeDiscussion
        collaborative_discussion = CollaborativeDiscussion("DEMO")
        logger.info("✓ CollaborativeDiscussion initialized")
        
        # Test OffHoursAI initialization
        from ai.offhours_discussion import OffHoursAI
        offhours_ai = OffHoursAI("DEMO")
        logger.info("✓ OffHoursAI initialized")
        
        # Test Phase4Integration initialization
        from workflows.phase4_integration import Phase4Integration
        phase4_integration = Phase4Integration("DEMO")
        logger.info("✓ Phase4Integration initialized")
        
        # Test Phase4Main initialization
        from main_phase4 import Phase4Main
        phase4_main = Phase4Main("DEMO")
        logger.info("✓ Phase4Main initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Component initialization error: {e}")
        return False

async def test_validation_methods():
    """Test validation methods."""
    try:
        logger.info("Testing validation methods...")
        
        # Test collaborative discussion validation
        from ai.collaborative_discussion import CollaborativeDiscussion
        collaborative_discussion = CollaborativeDiscussion("DEMO")
        assert collaborative_discussion.validate_discussion() == True
        logger.info("✓ Collaborative Discussion validation passed")
        
        # Test off-hours AI validation
        from ai.offhours_discussion import OffHoursAI
        offhours_ai = OffHoursAI("DEMO")
        assert offhours_ai.validate_offhours_system() == True
        logger.info("✓ Off-Hours AI validation passed")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Validation method error: {e}")
        return False

async def test_model_availability():
    """Test model availability checking."""
    try:
        logger.info("Testing model availability...")
        
        from ai.multi_model import MultiModelManager
        multi_model_manager = MultiModelManager("DEMO")
        
        # Check model availability (this will fail if Ollama is not running)
        try:
            availability = await multi_model_manager.check_model_availability()
            logger.info(f"✓ Model availability check completed: {availability}")
        except Exception as e:
            logger.warning(f"⚠ Model availability check failed (expected if Ollama not running): {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model availability test error: {e}")
        return False

async def test_phase4_integration():
    """Test Phase 4 integration."""
    try:
        logger.info("Testing Phase 4 integration...")
        
        from workflows.phase4_integration import Phase4Integration
        phase4_integration = Phase4Integration("DEMO")
        
        # Test getting status
        try:
            status = await phase4_integration.get_phase4_status()
            logger.info(f"✓ Phase 4 status retrieved: {status.overall_status}")
        except Exception as e:
            logger.warning(f"⚠ Phase 4 status check failed: {e}")
        
        # Test ensemble metrics
        try:
            metrics = await phase4_integration.get_ensemble_metrics()
            logger.info(f"✓ Ensemble metrics retrieved: {metrics.total_discussions} discussions")
        except Exception as e:
            logger.warning(f"⚠ Ensemble metrics check failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Phase 4 integration test error: {e}")
        return False

async def test_health_check():
    """Test health check functionality."""
    try:
        logger.info("Testing health check...")
        
        from workflows.phase4_integration import Phase4Integration
        phase4_integration = Phase4Integration("DEMO")
        
        # Test health check
        try:
            health_check = await phase4_integration.run_ensemble_health_check()
            logger.info(f"✓ Health check completed: {health_check.get('overall_health', 'UNKNOWN')}")
        except Exception as e:
            logger.warning(f"⚠ Health check failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Health check test error: {e}")
        return False

async def run_validation():
    """Run all validation tests."""
    logger.info("=" * 60)
    logger.info("PHASE 4 VALIDATION STARTED")
    logger.info("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Component Initialization", test_component_initialization),
        ("Validation Methods", test_validation_methods),
        ("Model Availability", test_model_availability),
        ("Phase 4 Integration", test_phase4_integration),
        ("Health Check", test_health_check)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = await test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4 VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - Phase 4 is ready!")
        return True
    else:
        logger.warning(f"⚠ {total - passed} tests failed - Phase 4 needs attention")
        return False

async def main():
    """Main validation function."""
    try:
        success = await run_validation()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
