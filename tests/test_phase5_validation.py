"""
Phase 5 Validation Test Script

This script validates the Phase 5 implementation including:
- Adaptive Configuration System
- Performance Learning Engine
- Self-Learning Engine
- Phase 5 Integration
"""

import logging
import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append('src')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test_phase5_imports():
    """Test Phase 5 imports."""
    logger.info("Testing Phase 5 imports...")
    
    try:
        # Test adaptive configuration imports
        from adaptive.configuration_manager import (
            get_adaptive_config_manager, PerformanceMetrics, ParameterType,
            AdjustmentTrigger, get_parameter, get_all_parameters
        )
        logger.info("✓ Adaptive Configuration imports successful")
        
        # Test performance learning imports
        from adaptive.performance_learning import (
            get_performance_learning_engine, TradeOutcome, IdentifiedPattern,
            LearningTrigger, PatternType, get_learning_summary
        )
        logger.info("✓ Performance Learning imports successful")
        
        # Test self-learning engine imports
        from adaptive.self_learning_engine import (
            get_self_learning_engine, OptimizationObjective, MetaParameterType,
            get_meta_parameter, get_all_meta_parameters, optimize_parameters
        )
        logger.info("✓ Self-Learning Engine imports successful")
        
        # Test Phase 5 integration imports
        from workflows.phase5_integration import (
            get_phase5_integration, initialize_phase5, start_adaptive_learning,
            get_phase5_status, get_optimization_summary
        )
        logger.info("✓ Phase 5 Integration imports successful")
        
        # Test Phase 5 main imports
        from main_phase5 import Phase5Main
        logger.info("✓ Phase 5 Main imports successful")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Import test failed: {e}")
        return False

async def test_adaptive_configuration():
    """Test adaptive configuration system."""
    logger.info("Testing Adaptive Configuration System...")
    
    try:
        from adaptive.configuration_manager import get_adaptive_config_manager, PerformanceMetrics
        
        # Initialize configuration manager
        config_manager = get_adaptive_config_manager("DEMO")
        logger.info("✓ Configuration Manager initialized")
        
        # Test parameter retrieval
        position_size = config_manager.get_parameter("position_size")
        all_params = config_manager.get_all_parameters()
        logger.info(f"✓ Retrieved parameters: position_size={position_size}, total_params={len(all_params)}")
        
        # Test performance metrics update
        test_metrics = PerformanceMetrics(
            win_rate=0.65,
            profit_factor=1.4,
            sharpe_ratio=0.8,
            max_drawdown=0.08,
            total_return=150.0,
            trade_count=25,
            avg_trade_duration=48.5,
            volatility=0.15,
            timestamp=datetime.now()
        )
        
        config_manager.update_performance_metrics(test_metrics)
        logger.info("✓ Performance metrics updated")
        
        # Test adjustment summary
        summary = config_manager.get_adjustment_summary()
        logger.info(f"✓ Adjustment summary: {summary}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Adaptive Configuration test failed: {e}")
        return False

async def test_performance_learning():
    """Test performance learning engine."""
    logger.info("Testing Performance Learning Engine...")
    
    try:
        from adaptive.performance_learning import (
            get_performance_learning_engine, TradeOutcome, LearningTrigger
        )
        
        # Initialize learning engine
        learning_engine = get_performance_learning_engine("DEMO")
        logger.info("✓ Learning Engine initialized")
        
        # Test learning summary
        summary = learning_engine.get_learning_summary()
        logger.info(f"✓ Learning summary: {summary}")
        
        # Test trade outcome recording (simulated)
        test_outcome = TradeOutcome(
            trade_id="TEST_001",
            symbol="AAPL",
            entry_time=datetime.now() - timedelta(hours=24),
            exit_time=datetime.now(),
            entry_price=150.0,
            exit_price=155.0,
            position_size=0.1,
            pnl=50.0,
            pnl_percentage=0.033,
            duration_hours=24.0,
            confidence_score=0.8,
            market_conditions={"volatility": 0.2, "trend": "bullish"},
            technical_indicators={"rsi": 65, "macd": 0.5},
            sentiment_scores={"overall": 0.7, "news": 0.8},
            risk_metrics={"volatility": 0.2, "beta": 1.1},
            exit_reason="take_profit",
            success=True
        )
        
        learning_engine.record_trade_outcome(test_outcome)
        logger.info("✓ Trade outcome recorded")
        
        # Test pattern retrieval
        recent_patterns = learning_engine.get_recent_patterns(days=7)
        logger.info(f"✓ Recent patterns: {len(recent_patterns)} found")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Performance Learning test failed: {e}")
        return False

async def test_self_learning_engine():
    """Test self-learning engine."""
    logger.info("Testing Self-Learning Engine...")
    
    try:
        from adaptive.self_learning_engine import (
            get_self_learning_engine, OptimizationObjective, MetaParameterType
        )
        
        # Initialize self-learning engine
        self_learning = get_self_learning_engine("DEMO")
        logger.info("✓ Self-Learning Engine initialized")
        
        # Test meta-parameter retrieval
        confidence_threshold = self_learning.get_meta_parameter("confidence_threshold")
        all_meta_params = self_learning.get_all_meta_parameters()
        logger.info(f"✓ Meta-parameters: confidence_threshold={confidence_threshold}, total={len(all_meta_params)}")
        
        # Test learning summary
        summary = self_learning.get_learning_summary()
        logger.info(f"✓ Learning summary: {summary['learning_state']['learning_phase']}")
        
        # Test optimization objective setting
        self_learning.set_learning_objective(OptimizationObjective.BALANCED_PERFORMANCE)
        logger.info("✓ Optimization objective set")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Self-Learning Engine test failed: {e}")
        return False

async def test_phase5_integration():
    """Test Phase 5 integration."""
    logger.info("Testing Phase 5 Integration...")
    
    try:
        from workflows.phase5_integration import (
            get_phase5_integration, initialize_phase5, get_phase5_status
        )
        
        # Initialize Phase 5 integration
        phase5 = get_phase5_integration("DEMO")
        logger.info("✓ Phase 5 Integration initialized")
        
        # Test initialization
        init_result = await initialize_phase5("DEMO")
        logger.info(f"✓ Phase 5 initialization: {init_result}")
        
        # Test status retrieval
        status = await get_phase5_status("DEMO")
        logger.info(f"✓ Phase 5 status: {status['phase']} - {status['mode']}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Phase 5 Integration test failed: {e}")
        return False

async def test_phase5_main():
    """Test Phase 5 main."""
    logger.info("Testing Phase 5 Main...")
    
    try:
        from main_phase5 import Phase5Main
        
        # Initialize Phase 5 main
        phase5_main = Phase5Main("DEMO")
        logger.info("✓ Phase 5 Main initialized")
        
        # Test initialization
        init_result = await phase5_main.initialize()
        logger.info(f"✓ Phase 5 Main initialization: {init_result}")
        
        # Test status retrieval
        status = await phase5_main.get_status()
        logger.info(f"✓ Phase 5 Main status: {status.get('phase', 'unknown')}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Phase 5 Main test failed: {e}")
        return False

async def test_optimization_workflow():
    """Test optimization workflow."""
    logger.info("Testing Optimization Workflow...")
    
    try:
        from adaptive.self_learning_engine import (
            get_self_learning_engine, OptimizationObjective
        )
        from adaptive.configuration_manager import get_adaptive_config_manager, PerformanceMetrics
        
        # Get engines
        self_learning = get_self_learning_engine("DEMO")
        config_manager = get_adaptive_config_manager("DEMO")
        
        # Add some performance data
        for i in range(5):
            metrics = PerformanceMetrics(
                win_rate=0.6 + (i * 0.02),
                profit_factor=1.2 + (i * 0.05),
                sharpe_ratio=0.5 + (i * 0.1),
                max_drawdown=0.1 - (i * 0.01),
                total_return=100.0 + (i * 20.0),
                trade_count=10 + i,
                avg_trade_duration=24.0 + i,
                volatility=0.2 - (i * 0.02),
                timestamp=datetime.now() - timedelta(days=5-i)
            )
            config_manager.update_performance_metrics(metrics)
        
        logger.info("✓ Performance data added")
        
        # Test optimization (this might take a moment)
        logger.info("Running optimization test...")
        result = await self_learning.optimize_parameters(OptimizationObjective.BALANCED_PERFORMANCE)
        logger.info(f"✓ Optimization completed: {result['status']}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Optimization workflow test failed: {e}")
        return False

async def run_all_tests():
    """Run all Phase 5 tests."""
    logger.info("=" * 60)
    logger.info("PHASE 5 VALIDATION STARTED")
    logger.info("=" * 60)
    
    tests = [
        ("Import Tests", test_phase5_imports),
        ("Adaptive Configuration", test_adaptive_configuration),
        ("Performance Learning", test_performance_learning),
        ("Self-Learning Engine", test_self_learning_engine),
        ("Phase 5 Integration", test_phase5_integration),
        ("Phase 5 Main", test_phase5_main),
        ("Optimization Workflow", test_optimization_workflow)
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
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 5 VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - Phase 5 is ready!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    asyncio.run(run_all_tests())
