"""
Integration Tests for Phase Integration
======================================

Tests the integration between all phases (1-5) and their
workflows and components.
"""

import pytest
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock

# Import phase components
from src.main_phase4 import Phase4Main
from src.main_phase5 import Phase5Main
from src.workflows.phase2_integration import get_phase2_integration
from src.workflows.phase3_integration import get_phase3_integration
from src.workflows.phase4_integration import get_phase4_integration
from src.workflows.phase5_integration import get_phase5_integration
from src.config.mode_manager import set_mode, get_current_mode

logger = logging.getLogger(__name__)


class TestPhaseIntegration:
    """Test suite for phase integration."""
    
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
    async def test_phase1_to_phase5_flow(self, setup_demo_mode, mock_market_data):
        """Test complete flow from Phase 1 to Phase 5."""
        logger.info("Testing Phase 1 to Phase 5 flow...")
        
        # Initialize Phase 4 (includes all previous phases)
        phase4_main = Phase4Main("DEMO")
        await phase4_main.initialize()
        
        # Initialize Phase 5
        phase5_main = Phase5Main("DEMO")
        await phase5_main.initialize()
        
        # Test Phase 2 integration
        phase2_integration = get_phase2_integration()
        phase2_status = await phase2_integration.get_system_status()
        assert phase2_status is not None
        assert "trading_cycle" in phase2_status
        assert "activity_scheduler" in phase2_status
        
        # Test Phase 3 integration
        phase3_integration = get_phase3_integration()
        phase3_status = await phase3_integration.get_system_status()
        assert phase3_status is not None
        assert "advanced_models" in phase3_status
        assert "autonomous_trading" in phase3_status
        
        # Test Phase 4 integration
        phase4_integration = get_phase4_integration()
        phase4_status = await phase4_integration.get_system_status()
        assert phase4_status is not None
        assert "multi_model" in phase4_status
        assert "collaborative_discussion" in phase4_status
        
        # Test Phase 5 integration
        phase5_integration = get_phase5_integration()
        phase5_status = await phase5_integration.get_system_status()
        assert phase5_status is not None
        assert "adaptive_config" in phase5_status
        assert "performance_learning" in phase5_status
        
        logger.info("✓ Phase 1 to Phase 5 flow completed successfully")
    
    @pytest.mark.asyncio
    async def test_adaptive_learning_with_trading(self, setup_demo_mode, mock_market_data):
        """Test adaptive learning integration with trading operations."""
        logger.info("Testing adaptive learning with trading...")
        
        # Get Phase 5 integration
        phase5_integration = get_phase5_integration()
        
        # Simulate trading activity
        trading_results = [
            {
                "symbol": "AAPL",
                "entry_price": 150.0,
                "exit_price": 155.0,
                "pnl": 50.0,
                "confidence": 0.85,
                "timestamp": datetime.now()
            },
            {
                "symbol": "MSFT",
                "entry_price": 300.0,
                "exit_price": 295.0,
                "pnl": -25.0,
                "confidence": 0.70,
                "timestamp": datetime.now()
            }
        ]
        
        # Test performance learning
        for result in trading_results:
            learning_result = await phase5_integration.learn_from_trade_result(result)
            assert learning_result is not None
            assert "parameter_adjustments" in learning_result
        
        # Test adaptive configuration
        config_result = await phase5_integration.get_adaptive_configuration()
        assert config_result is not None
        assert "current_parameters" in config_result
        assert "learning_metrics" in config_result
        
        # Test self-learning engine
        learning_status = await phase5_integration.get_learning_status()
        assert learning_status is not None
        assert "learning_active" in learning_status
        assert "optimization_progress" in learning_status
        
        logger.info("✓ Adaptive learning with trading completed successfully")
    
    @pytest.mark.asyncio
    async def test_multi_model_collaboration(self, setup_demo_mode, mock_market_data):
        """Test multi-model collaboration integration."""
        logger.info("Testing multi-model collaboration...")
        
        # Get Phase 4 integration
        phase4_integration = get_phase4_integration()
        
        # Test multi-model analysis
        analysis_result = await phase4_integration.analyze_with_ensemble(
            symbols=["AAPL", "MSFT"],
            market_data=mock_market_data
        )
        
        assert analysis_result is not None
        assert "ensemble_analysis" in analysis_result
        assert "model_opinions" in analysis_result
        assert "final_decision" in analysis_result
        
        # Verify multiple models participated
        model_opinions = analysis_result["model_opinions"]
        assert len(model_opinions) > 0
        
        # Test collaborative discussion
        discussion_result = await phase4_integration.run_collaborative_discussion(
            topic="Market Analysis",
            context={"symbols": ["AAPL"], "market_data": mock_market_data}
        )
        
        assert discussion_result is not None
        assert "discussion_rounds" in discussion_result
        assert "final_consensus" in discussion_result
        
        # Test off-hours discussion
        offhours_result = await phase4_integration.run_offhours_analysis()
        assert offhours_result is not None
        assert "insights" in offhours_result
        assert "improvements" in offhours_result
        
        logger.info("✓ Multi-model collaboration completed successfully")
    
    @pytest.mark.asyncio
    async def test_background_learning_processes(self, setup_demo_mode):
        """Test background learning processes integration."""
        logger.info("Testing background learning processes...")
        
        # Get Phase 5 integration
        phase5_integration = get_phase5_integration()
        
        # Test background learning status
        bg_status = await phase5_integration.get_background_learning_status()
        assert bg_status is not None
        assert "processes_active" in bg_status
        assert "learning_sessions" in bg_status
        
        # Test performance learning background process
        perf_learning = await phase5_integration.get_performance_learning_status()
        assert perf_learning is not None
        assert "learning_active" in perf_learning
        assert "trade_analysis_count" in perf_learning
        
        # Test self-learning background process
        self_learning = await phase5_integration.get_self_learning_status()
        assert self_learning is not None
        assert "optimization_active" in self_learning
        assert "parameter_updates" in self_learning
        
        # Test adaptive configuration background process
        adaptive_config = await phase5_integration.get_adaptive_config_status()
        assert adaptive_config is not None
        assert "monitoring_active" in adaptive_config
        assert "parameter_adjustments" in adaptive_config
        
        logger.info("✓ Background learning processes completed successfully")
    
    @pytest.mark.asyncio
    async def test_parameter_optimization_integration(self, setup_demo_mode, mock_market_data):
        """Test parameter optimization integration across phases."""
        logger.info("Testing parameter optimization integration...")
        
        # Get Phase 5 integration
        phase5_integration = get_phase5_integration()
        
        # Test parameter optimization
        optimization_result = await phase5_integration.optimize_parameters(
            optimization_type="performance_based"
        )
        
        assert optimization_result is not None
        assert "optimization_completed" in optimization_result
        assert "parameter_changes" in optimization_result
        assert "performance_improvement" in optimization_result
        
        # Test meta-parameter optimization
        meta_optimization = await phase5_integration.optimize_meta_parameters()
        assert meta_optimization is not None
        assert "meta_optimization_completed" in meta_optimization
        assert "meta_parameter_changes" in meta_optimization
        
        # Test adaptive parameter adjustment
        adaptive_result = await phase5_integration.adjust_parameters_adaptively()
        assert adaptive_result is not None
        assert "adjustments_made" in adaptive_result
        assert "adjustment_reasons" in adaptive_result
        
        # Test parameter validation
        validation_result = await phase5_integration.validate_parameter_changes()
        assert validation_result is not None
        assert "validation_passed" in validation_result
        assert "validation_details" in validation_result
        
        logger.info("✓ Parameter optimization integration completed successfully")
    
    @pytest.mark.asyncio
    async def test_cross_phase_data_flow(self, setup_demo_mode, mock_market_data):
        """Test data flow between different phases."""
        logger.info("Testing cross-phase data flow...")
        
        # Test data flow from Phase 2 to Phase 3
        phase2_integration = get_phase2_integration()
        phase3_integration = get_phase3_integration()
        
        # Get trading cycle data
        trading_data = await phase2_integration.get_trading_data()
        assert trading_data is not None
        
        # Pass to Phase 3 for advanced analysis
        advanced_analysis = await phase3_integration.analyze_with_advanced_models(
            trading_data=trading_data,
            market_data=mock_market_data
        )
        assert advanced_analysis is not None
        
        # Test data flow from Phase 3 to Phase 4
        phase4_integration = get_phase4_integration()
        
        # Pass advanced analysis to Phase 4 for ensemble analysis
        ensemble_analysis = await phase4_integration.analyze_with_ensemble(
            symbols=["AAPL"],
            market_data=mock_market_data,
            previous_analysis=advanced_analysis
        )
        assert ensemble_analysis is not None
        
        # Test data flow from Phase 4 to Phase 5
        phase5_integration = get_phase5_integration()
        
        # Pass ensemble analysis to Phase 5 for learning
        learning_result = await phase5_integration.learn_from_analysis(
            analysis_result=ensemble_analysis,
            market_data=mock_market_data
        )
        assert learning_result is not None
        
        logger.info("✓ Cross-phase data flow completed successfully")
    
    @pytest.mark.asyncio
    async def test_phase_coordination(self, setup_demo_mode):
        """Test coordination between all phases."""
        logger.info("Testing phase coordination...")
        
        # Test Phase 2 coordination
        phase2_integration = get_phase2_integration()
        phase2_coordination = await phase2_integration.coordinate_phase_activities()
        assert phase2_coordination is not None
        assert "coordination_status" in phase2_coordination
        
        # Test Phase 3 coordination
        phase3_integration = get_phase3_integration()
        phase3_coordination = await phase3_integration.coordinate_phase_activities()
        assert phase3_coordination is not None
        assert "coordination_status" in phase3_coordination
        
        # Test Phase 4 coordination
        phase4_integration = get_phase4_integration()
        phase4_coordination = await phase4_integration.coordinate_phase_activities()
        assert phase4_coordination is not None
        assert "coordination_status" in phase4_coordination
        
        # Test Phase 5 coordination
        phase5_integration = get_phase5_integration()
        phase5_coordination = await phase5_integration.coordinate_phase_activities()
        assert phase5_coordination is not None
        assert "coordination_status" in phase5_coordination
        
        # Test overall system coordination
        overall_coordination = await phase5_integration.coordinate_all_phases()
        assert overall_coordination is not None
        assert "all_phases_coordinated" in overall_coordination
        assert "coordination_summary" in overall_coordination
        
        logger.info("✓ Phase coordination completed successfully")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
