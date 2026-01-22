"""
Validation Tests for Phase 1-5
==============================

Comprehensive validation tests for all phases (1-5) to ensure
system readiness and integration.
"""

import pytest
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock

# Import all phase components
from src.main import UnifiedTradingSystem
from src.main_phase4 import Phase4Main
from src.main_phase5 import Phase5Main
from src.config.mode_manager import set_mode, get_current_mode
from src.workflows.phase2_integration import get_phase2_integration
from src.workflows.phase3_integration import get_phase3_integration
from src.workflows.phase4_integration import get_phase4_integration
from src.workflows.phase5_integration import get_phase5_integration

logger = logging.getLogger(__name__)


class TestPhase1to5Validation:
    """Test suite for Phase 1-5 validation."""
    
    @pytest.fixture
    async def setup_demo_mode(self):
        """Setup demo mode for testing."""
        set_mode("DEMO")
        yield
        # Cleanup after test
    
    @pytest.fixture
    async def setup_unified_system(self):
        """Setup unified trading system for testing."""
        system = UnifiedTradingSystem("DEMO")
        yield system
    
    @pytest.mark.asyncio
    async def test_all_phases_initialized(self, setup_demo_mode, setup_unified_system):
        """Test that all phases are properly initialized."""
        logger.info("Testing all phases initialization...")
        
        system = setup_unified_system
        
        # Test system initialization
        init_result = await system.initialize_system()
        assert init_result is True, "Unified system initialization should succeed"
        
        # Test Phase 4 initialization
        phase4_main = Phase4Main("DEMO")
        phase4_init_result = await phase4_main.initialize()
        assert phase4_init_result is True, "Phase 4 initialization should succeed"
        
        # Test Phase 5 initialization
        phase5_main = Phase5Main("DEMO")
        phase5_init_result = await phase5_main.initialize()
        assert phase5_init_result is True, "Phase 5 initialization should succeed"
        
        # Test phase integration components
        phase2_integration = get_phase2_integration()
        assert phase2_integration is not None, "Phase 2 integration should be available"
        
        phase3_integration = get_phase3_integration()
        assert phase3_integration is not None, "Phase 3 integration should be available"
        
        phase4_integration = get_phase4_integration()
        assert phase4_integration is not None, "Phase 4 integration should be available"
        
        phase5_integration = get_phase5_integration()
        assert phase5_integration is not None, "Phase 5 integration should be available"
        
        logger.info("✓ All phases initialized successfully")
    
    @pytest.mark.asyncio
    async def test_all_components_operational(self, setup_demo_mode, setup_unified_system):
        """Test that all components are operational."""
        logger.info("Testing all components operational status...")
        
        system = setup_unified_system
        await system.initialize_system()
        
        # Test system status
        system_status = await system.get_system_status()
        assert system_status is not None, "System status should be available"
        assert "timestamp" in system_status, "System status should have timestamp"
        assert "mode" in system_status, "System status should have mode"
        assert system_status["mode"] == "DEMO", "System should be in DEMO mode"
        
        # Test Phase 2 components
        phase2_integration = get_phase2_integration()
        phase2_status = await phase2_integration.get_system_status()
        assert phase2_status is not None, "Phase 2 status should be available"
        assert "trading_cycle" in phase2_status, "Phase 2 should have trading cycle"
        assert "activity_scheduler" in phase2_status, "Phase 2 should have activity scheduler"
        
        # Test Phase 3 components
        phase3_integration = get_phase3_integration()
        phase3_status = await phase3_integration.get_system_status()
        assert phase3_status is not None, "Phase 3 status should be available"
        assert "advanced_models" in phase3_status, "Phase 3 should have advanced models"
        assert "autonomous_trading" in phase3_status, "Phase 3 should have autonomous trading"
        
        # Test Phase 4 components
        phase4_integration = get_phase4_integration()
        phase4_status = await phase4_integration.get_system_status()
        assert phase4_status is not None, "Phase 4 status should be available"
        assert "multi_model" in phase4_status, "Phase 4 should have multi-model"
        assert "collaborative_discussion" in phase4_status, "Phase 4 should have collaborative discussion"
        
        # Test Phase 5 components
        phase5_integration = get_phase5_integration()
        phase5_status = await phase5_integration.get_system_status()
        assert phase5_status is not None, "Phase 5 status should be available"
        assert "adaptive_config" in phase5_status, "Phase 5 should have adaptive config"
        assert "performance_learning" in phase5_status, "Phase 5 should have performance learning"
        
        logger.info("✓ All components are operational")
    
    @pytest.mark.asyncio
    async def test_integration_working(self, setup_demo_mode, setup_unified_system):
        """Test that all integrations are working properly."""
        logger.info("Testing integration functionality...")
        
        system = setup_unified_system
        await system.initialize_system()
        
        # Test Phase 2 integration
        phase2_integration = get_phase2_integration()
        phase2_coordination = await phase2_integration.coordinate_phase_activities()
        assert phase2_coordination is not None, "Phase 2 coordination should work"
        assert "coordination_status" in phase2_coordination, "Phase 2 should have coordination status"
        
        # Test Phase 3 integration
        phase3_integration = get_phase3_integration()
        phase3_coordination = await phase3_integration.coordinate_phase_activities()
        assert phase3_coordination is not None, "Phase 3 coordination should work"
        assert "coordination_status" in phase3_coordination, "Phase 3 should have coordination status"
        
        # Test Phase 4 integration
        phase4_integration = get_phase4_integration()
        phase4_coordination = await phase4_integration.coordinate_phase_activities()
        assert phase4_coordination is not None, "Phase 4 coordination should work"
        assert "coordination_status" in phase4_coordination, "Phase 4 should have coordination status"
        
        # Test Phase 5 integration
        phase5_integration = get_phase5_integration()
        phase5_coordination = await phase5_integration.coordinate_phase_activities()
        assert phase5_coordination is not None, "Phase 5 coordination should work"
        assert "coordination_status" in phase5_coordination, "Phase 5 should have coordination status"
        
        # Test overall system coordination
        overall_coordination = await phase5_integration.coordinate_all_phases()
        assert overall_coordination is not None, "Overall coordination should work"
        assert "all_phases_coordinated" in overall_coordination, "Should have overall coordination status"
        
        logger.info("✓ All integrations are working properly")
    
    @pytest.mark.asyncio
    async def test_no_critical_errors(self, setup_demo_mode, setup_unified_system):
        """Test that there are no critical errors in the system."""
        logger.info("Testing for critical errors...")
        
        system = setup_unified_system
        
        # Test system initialization without errors
        try:
            init_result = await system.initialize_system()
            assert init_result is True, "System initialization should succeed without errors"
        except Exception as e:
            pytest.fail(f"System initialization failed with error: {e}")
        
        # Test system startup without errors
        try:
            start_result = await system.start_system()
            assert start_result is True, "System startup should succeed without errors"
        except Exception as e:
            pytest.fail(f"System startup failed with error: {e}")
        
        # Test system status without errors
        try:
            system_status = await system.get_system_status()
            assert system_status is not None, "System status should be available without errors"
        except Exception as e:
            pytest.fail(f"System status retrieval failed with error: {e}")
        
        # Test phase execution without errors
        try:
            phase_result = await system.run_phase("4")
            assert phase_result is True, "Phase execution should succeed without errors"
        except Exception as e:
            pytest.fail(f"Phase execution failed with error: {e}")
        
        # Test system shutdown without errors
        try:
            stop_result = await system.stop_system()
            assert stop_result is True, "System shutdown should succeed without errors"
        except Exception as e:
            pytest.fail(f"System shutdown failed with error: {e}")
        
        logger.info("✓ No critical errors found in the system")
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, setup_demo_mode, setup_unified_system):
        """Test that system meets performance benchmarks."""
        logger.info("Testing performance benchmarks...")
        
        system = setup_unified_system
        
        # Test initialization performance
        start_time = asyncio.get_event_loop().time()
        init_result = await system.initialize_system()
        init_time = asyncio.get_event_loop().time() - start_time
        
        assert init_result is True, "Initialization should succeed"
        assert init_time < 10.0, f"Initialization should take less than 10s, took {init_time:.2f}s"
        
        # Test startup performance
        start_time = asyncio.get_event_loop().time()
        start_result = await system.start_system()
        start_time_elapsed = asyncio.get_event_loop().time() - start_time
        
        assert start_result is True, "Startup should succeed"
        assert start_time_elapsed < 5.0, f"Startup should take less than 5s, took {start_time_elapsed:.2f}s"
        
        # Test status retrieval performance
        start_time = asyncio.get_event_loop().time()
        system_status = await system.get_system_status()
        status_time = asyncio.get_event_loop().time() - start_time
        
        assert system_status is not None, "Status retrieval should succeed"
        assert status_time < 1.0, f"Status retrieval should take less than 1s, took {status_time:.2f}s"
        
        # Test phase execution performance
        start_time = asyncio.get_event_loop().time()
        phase_result = await system.run_phase("4")
        phase_time = asyncio.get_event_loop().time() - start_time
        
        assert phase_result is True, "Phase execution should succeed"
        assert phase_time < 15.0, f"Phase execution should take less than 15s, took {phase_time:.2f}s"
        
        # Test shutdown performance
        start_time = asyncio.get_event_loop().time()
        stop_result = await system.stop_system()
        stop_time = asyncio.get_event_loop().time() - start_time
        
        assert stop_result is True, "Shutdown should succeed"
        assert stop_time < 3.0, f"Shutdown should take less than 3s, took {stop_time:.2f}s"
        
        logger.info(f"✓ Performance benchmarks met: Init {init_time:.2f}s, Start {start_time_elapsed:.2f}s, Status {status_time:.2f}s, Phase {phase_time:.2f}s, Stop {stop_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_mode_switching(self, setup_demo_mode, setup_unified_system):
        """Test mode switching functionality."""
        logger.info("Testing mode switching...")
        
        system = setup_unified_system
        await system.initialize_system()
        
        # Test DEMO mode
        set_mode("DEMO")
        assert get_current_mode() == "DEMO", "Should be in DEMO mode"
        
        # Test LIVE mode
        set_mode("LIVE")
        assert get_current_mode() == "LIVE", "Should be in LIVE mode"
        
        # Test back to DEMO mode
        set_mode("DEMO")
        assert get_current_mode() == "DEMO", "Should be back in DEMO mode"
        
        # Test system status in different modes
        system_status = await system.get_system_status()
        assert system_status is not None, "System status should be available in any mode"
        assert system_status["mode"] == "DEMO", "System status should reflect current mode"
        
        logger.info("✓ Mode switching functionality works correctly")
    
    @pytest.mark.asyncio
    async def test_phase_coordination(self, setup_demo_mode, setup_unified_system):
        """Test phase coordination and communication."""
        logger.info("Testing phase coordination...")
        
        system = setup_unified_system
        await system.initialize_system()
        
        # Test individual phase coordination
        phase2_integration = get_phase2_integration()
        phase3_integration = get_phase3_integration()
        phase4_integration = get_phase4_integration()
        phase5_integration = get_phase5_integration()
        
        # Test Phase 2 coordination
        phase2_coordination = await phase2_integration.coordinate_phase_activities()
        assert phase2_coordination is not None, "Phase 2 coordination should work"
        
        # Test Phase 3 coordination
        phase3_coordination = await phase3_integration.coordinate_phase_activities()
        assert phase3_coordination is not None, "Phase 3 coordination should work"
        
        # Test Phase 4 coordination
        phase4_coordination = await phase4_integration.coordinate_phase_activities()
        assert phase4_coordination is not None, "Phase 4 coordination should work"
        
        # Test Phase 5 coordination
        phase5_coordination = await phase5_integration.coordinate_phase_activities()
        assert phase5_coordination is not None, "Phase 5 coordination should work"
        
        # Test overall coordination
        overall_coordination = await phase5_integration.coordinate_all_phases()
        assert overall_coordination is not None, "Overall coordination should work"
        assert "all_phases_coordinated" in overall_coordination, "Should have overall coordination status"
        
        logger.info("✓ Phase coordination works correctly")
    
    @pytest.mark.asyncio
    async def test_system_reliability(self, setup_demo_mode, setup_unified_system):
        """Test system reliability and stability."""
        logger.info("Testing system reliability...")
        
        system = setup_unified_system
        
        # Test multiple initialization cycles
        for cycle in range(3):
            try:
                # Initialize system
                init_result = await system.initialize_system()
                assert init_result is True, f"Initialization cycle {cycle + 1} should succeed"
                
                # Start system
                start_result = await system.start_system()
                assert start_result is True, f"Startup cycle {cycle + 1} should succeed"
                
                # Get system status
                system_status = await system.get_system_status()
                assert system_status is not None, f"Status retrieval cycle {cycle + 1} should succeed"
                
                # Stop system
                stop_result = await system.stop_system()
                assert stop_result is True, f"Shutdown cycle {cycle + 1} should succeed"
                
                logger.info(f"✓ Reliability cycle {cycle + 1} completed successfully")
                
            except Exception as e:
                pytest.fail(f"Reliability cycle {cycle + 1} failed: {e}")
        
        logger.info("✓ System reliability tests passed")
    
    @pytest.mark.asyncio
    async def test_component_integration(self, setup_demo_mode, setup_unified_system):
        """Test component integration and data flow."""
        logger.info("Testing component integration...")
        
        system = setup_unified_system
        await system.initialize_system()
        
        # Test Phase 2 to Phase 3 data flow
        phase2_integration = get_phase2_integration()
        phase3_integration = get_phase3_integration()
        
        trading_data = await phase2_integration.get_trading_data()
        assert trading_data is not None, "Phase 2 should provide trading data"
        
        # Test Phase 3 to Phase 4 data flow
        phase4_integration = get_phase4_integration()
        
        # Mock market data for testing
        mock_market_data = {
            "AAPL": {
                "price": 150.0,
                "volume": 1000000,
                "timestamp": datetime.now()
            }
        }
        
        # Test Phase 4 to Phase 5 data flow
        phase5_integration = get_phase5_integration()
        
        # Test adaptive configuration
        adaptive_config = await phase5_integration.get_adaptive_configuration()
        assert adaptive_config is not None, "Phase 5 should provide adaptive configuration"
        assert "current_parameters" in adaptive_config, "Should have current parameters"
        
        # Test performance learning
        learning_status = await phase5_integration.get_learning_status()
        assert learning_status is not None, "Phase 5 should provide learning status"
        assert "learning_active" in learning_status, "Should have learning status"
        
        logger.info("✓ Component integration works correctly")
    
    @pytest.mark.asyncio
    async def test_system_monitoring(self, setup_demo_mode, setup_unified_system):
        """Test system monitoring capabilities."""
        logger.info("Testing system monitoring...")
        
        system = setup_unified_system
        await system.initialize_system()
        
        # Test system status monitoring
        system_status = await system.get_system_status()
        assert system_status is not None, "System status should be available"
        assert "timestamp" in system_status, "Should have timestamp"
        assert "mode" in system_status, "Should have mode"
        assert "is_running" in system_status, "Should have running status"
        assert "phases" in system_status, "Should have phases information"
        
        # Test validation monitoring
        if "validation" in system_status:
            validation_status = system_status["validation"]
            assert "security" in validation_status, "Should have security validation"
            assert "hallucination" in validation_status, "Should have hallucination validation"
            assert "debug" in validation_status, "Should have debug validation"
        
        # Test phase-specific monitoring
        phase5_integration = get_phase5_integration()
        phase5_status = await phase5_integration.get_system_status()
        assert phase5_status is not None, "Phase 5 status should be available"
        assert "adaptive_config" in phase5_status, "Should have adaptive config status"
        assert "performance_learning" in phase5_status, "Should have performance learning status"
        
        logger.info("✓ System monitoring works correctly")
    
    @pytest.mark.asyncio
    async def test_error_handling(self, setup_demo_mode, setup_unified_system):
        """Test error handling and recovery."""
        logger.info("Testing error handling...")
        
        system = setup_unified_system
        
        # Test error handling during initialization
        try:
            init_result = await system.initialize_system()
            assert init_result is True, "Initialization should handle errors gracefully"
        except Exception as e:
            # If initialization fails, it should be a known error type
            assert isinstance(e, (ValueError, RuntimeError, ConnectionError)), f"Unexpected error type: {type(e)}"
        
        # Test error handling during startup
        try:
            start_result = await system.start_system()
            assert start_result is True, "Startup should handle errors gracefully"
        except Exception as e:
            # If startup fails, it should be a known error type
            assert isinstance(e, (ValueError, RuntimeError, ConnectionError)), f"Unexpected error type: {type(e)}"
        
        # Test error handling during phase execution
        try:
            phase_result = await system.run_phase("4")
            assert phase_result is True, "Phase execution should handle errors gracefully"
        except Exception as e:
            # If phase execution fails, it should be a known error type
            assert isinstance(e, (ValueError, RuntimeError, ConnectionError)), f"Unexpected error type: {type(e)}"
        
        # Test error handling during shutdown
        try:
            stop_result = await system.stop_system()
            assert stop_result is True, "Shutdown should handle errors gracefully"
        except Exception as e:
            # If shutdown fails, it should be a known error type
            assert isinstance(e, (ValueError, RuntimeError, ConnectionError)), f"Unexpected error type: {type(e)}"
        
        logger.info("✓ Error handling works correctly")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
