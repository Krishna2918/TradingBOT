"""
Test Suite for Phase 4: Collaborative AI Ensemble

This module provides comprehensive tests for all Phase 4 components.
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Import Phase 4 components
from src.ai.multi_model import MultiModelManager, ModelRole, ModelOpinion
from src.ai.collaborative_discussion import CollaborativeDiscussion, DiscussionSession, DiscussionRound, FinalDecision
from src.ai.offhours_discussion import OffHoursAI, OffHoursSession, DiscussionTopic, Insight, Pattern, Improvement
from src.workflows.phase4_integration import Phase4Integration, Phase4Status, CollaborativeDecision, AIEnsembleMetrics
from src.main_phase4 import Phase4Main

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestMultiModelManager:
    """Test Multi-Model Manager."""
    
    @pytest.fixture
    def mock_ollama_client(self):
        """Mock Ollama client."""
        client = Mock()
        client.generate = AsyncMock()
        client.chat = AsyncMock()
        return client
    
    @pytest.fixture
    def multi_model_manager(self, mock_ollama_client):
        """Create MultiModelManager instance."""
        with patch('src.ai.multi_model.OllamaClient', return_value=mock_ollama_client):
            return MultiModelManager("DEMO")
    
    @pytest.mark.asyncio
    async def test_model_availability_check(self, multi_model_manager, mock_ollama_client):
        """Test model availability checking."""
        # Mock successful model check
        mock_ollama_client.generate.return_value = "Test response"
        
        availability = await multi_model_manager.check_model_availability()
        
        assert isinstance(availability, dict)
        assert len(availability) > 0
        # Should have at least one model available
        assert any(availability.values())
    
    @pytest.mark.asyncio
    async def test_get_model_opinion(self, multi_model_manager, mock_ollama_client):
        """Test getting model opinion."""
        # Mock model response
        mock_ollama_client.generate.return_value = "BUY: Strong bullish momentum detected"
        
        opinion = await multi_model_manager.get_model_opinion("qwen3-coder:480b-cloud", "BUY_OR_SELL", {"price": 100})
        
        assert isinstance(opinion, ModelOpinion)
        assert opinion.model_name == "qwen3-coder:480b-cloud"
        assert opinion.topic == "BUY_OR_SELL"
        assert opinion.opinion is not None
        assert opinion.confidence >= 0.0
        assert opinion.confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_get_all_model_opinions(self, multi_model_manager, mock_ollama_client):
        """Test getting all model opinions."""
        # Mock model responses
        mock_ollama_client.generate.return_value = "BUY: Strong bullish momentum detected"
        
        opinions = await multi_model_manager.get_all_model_opinions("BUY_OR_SELL", {"price": 100})
        
        assert isinstance(opinions, list)
        assert len(opinions) > 0
        for opinion in opinions:
            assert isinstance(opinion, ModelOpinion)
            assert opinion.topic == "BUY_OR_SELL"

class TestCollaborativeDiscussion:
    """Test Collaborative Discussion Framework."""
    
    @pytest.fixture
    def collaborative_discussion(self):
        """Create CollaborativeDiscussion instance."""
        return CollaborativeDiscussion("DEMO")
    
    def test_validate_discussion(self, collaborative_discussion):
        """Test discussion validation."""
        assert collaborative_discussion.validate_discussion() == True
    
    @pytest.mark.asyncio
    async def test_conduct_discussion(self, collaborative_discussion):
        """Test conducting discussion."""
        # Mock market data
        market_data = {
            "symbol": "AAPL",
            "price": 150.0,
            "volume": 1000000,
            "sentiment": 0.7
        }
        
        # Mock model opinions
        with patch('src.ai.collaborative_discussion.get_all_model_opinions') as mock_opinions:
            mock_opinions.return_value = [
                ModelOpinion(
                    model_name="qwen3-coder:480b-cloud",
                    topic="BUY_OR_SELL",
                    opinion="BUY: Strong bullish momentum",
                    confidence=0.8,
                    reasoning="Technical analysis shows upward trend",
                    timestamp=datetime.now()
                ),
                ModelOpinion(
                    model_name="deepseek-v3.1:671b-cloud",
                    topic="BUY_OR_SELL",
                    opinion="BUY: Positive sentiment",
                    confidence=0.7,
                    reasoning="Sentiment analysis is positive",
                    timestamp=datetime.now()
                )
            ]
            
            session = await collaborative_discussion.conduct_discussion("BUY_OR_SELL", market_data)
            
            assert isinstance(session, DiscussionSession)
            assert session.topic == "BUY_OR_SELL"
            assert len(session.rounds) > 0
            assert session.final_decision is not None
            assert session.final_decision.confidence > 0.0

class TestOffHoursAI:
    """Test Off-Hours AI System."""
    
    @pytest.fixture
    def offhours_ai(self):
        """Create OffHoursAI instance."""
        return OffHoursAI("DEMO")
    
    def test_validate_offhours_system(self, offhours_ai):
        """Test off-hours system validation."""
        assert offhours_ai.validate_offhours_system() == True
    
    @pytest.mark.asyncio
    async def test_conduct_offhours_analysis(self, offhours_ai):
        """Test conducting off-hours analysis."""
        # Mock model responses
        with patch('src.ai.offhours_discussion.get_all_model_opinions') as mock_opinions:
            mock_opinions.return_value = [
                ModelOpinion(
                    model_name="qwen3-coder:480b-cloud",
                    topic="TRADE_ANALYSIS",
                    opinion="Trade was successful",
                    confidence=0.8,
                    reasoning="Good entry and exit points",
                    timestamp=datetime.now()
                )
            ]
            
            session = await offhours_ai.conduct_offhours_analysis(DiscussionTopic.TRADE_ANALYSIS)
            
            assert isinstance(session, OffHoursSession)
            assert session.topic == DiscussionTopic.TRADE_ANALYSIS
            assert len(session.insights) > 0
            assert len(session.patterns) > 0
            assert len(session.improvements) > 0

class TestPhase4Integration:
    """Test Phase 4 Integration."""
    
    @pytest.fixture
    def phase4_integration(self):
        """Create Phase4Integration instance."""
        return Phase4Integration("DEMO")
    
    @pytest.mark.asyncio
    async def test_initialize_phase4(self, phase4_integration):
        """Test Phase 4 initialization."""
        # Mock component initialization
        with patch.object(phase4_integration.multi_model_manager, 'check_model_availability') as mock_availability:
            mock_availability.return_value = {"qwen3-coder:480b-cloud": True}
            
            with patch.object(phase4_integration.collaborative_discussion, 'validate_discussion') as mock_discussion:
                mock_discussion.return_value = True
                
                with patch.object(phase4_integration.offhours_ai, 'validate_offhours_system') as mock_offhours:
                    mock_offhours.return_value = True
                    
                    with patch.object(phase4_integration.phase3_integration, 'validate_phase3_integration') as mock_phase3:
                        mock_phase3.return_value = True
                        
                        result = await phase4_integration.initialize_phase4()
                        assert result == True
                        assert phase4_integration.is_initialized == True
    
    @pytest.mark.asyncio
    async def test_get_phase4_status(self, phase4_integration):
        """Test getting Phase 4 status."""
        # Mock component status
        with patch.object(phase4_integration.multi_model_manager, 'check_model_availability') as mock_availability:
            mock_availability.return_value = {"qwen3-coder:480b-cloud": True}
            
            with patch.object(phase4_integration.collaborative_discussion, 'validate_discussion') as mock_discussion:
                mock_discussion.return_value = True
                
                with patch.object(phase4_integration.offhours_ai, 'validate_offhours_system') as mock_offhours:
                    mock_offhours.return_value = True
                    
                    with patch.object(phase4_integration.phase3_integration, 'get_phase3_status') as mock_phase3:
                        mock_phase3.return_value = Mock(overall_status="FULLY_OPERATIONAL")
                        
                        status = await phase4_integration.get_phase4_status()
                        
                        assert isinstance(status, Phase4Status)
                        assert status.multi_model_integration == True
                        assert status.collaborative_discussion == True
                        assert status.offhours_discussion == True
                        assert status.phase3_integration == True
                        assert status.overall_status == "FULLY_OPERATIONAL"
    
    @pytest.mark.asyncio
    async def test_make_collaborative_decision(self, phase4_integration):
        """Test making collaborative decision."""
        # Mock market data
        market_data = {
            "symbol": "AAPL",
            "price": 150.0,
            "volume": 1000000,
            "sentiment": 0.7
        }
        
        # Mock collaborative discussion
        with patch('src.ai.collaborative_discussion.conduct_discussion') as mock_discussion:
            mock_session = Mock()
            mock_session.final_decision = Mock()
            mock_session.final_decision.recommendation = "BUY"
            mock_session.final_decision.confidence = 0.8
            mock_session.final_decision.reasoning = "Strong bullish momentum"
            mock_session.final_decision.model_contributions = {"qwen3-coder:480b-cloud": 0.8}
            mock_session.final_decision.discussion_summary = "Models agree on buy"
            mock_session.final_decision.timestamp = datetime.now()
            mock_discussion.return_value = mock_session
            
            decision = await phase4_integration.make_collaborative_decision("BUY_OR_SELL", market_data)
            
            assert isinstance(decision, CollaborativeDecision)
            assert decision.topic == "BUY_OR_SELL"
            assert decision.recommendation == "BUY"
            assert decision.confidence == 0.8
            assert decision.reasoning == "Strong bullish momentum"
    
    @pytest.mark.asyncio
    async def test_conduct_offhours_analysis_session(self, phase4_integration):
        """Test conducting off-hours analysis session."""
        # Mock off-hours analysis
        with patch('src.ai.offhours_discussion.conduct_offhours_analysis') as mock_analysis:
            mock_session = Mock()
            mock_session.session_id = "test-session-123"
            mock_session.topic = DiscussionTopic.TRADE_ANALYSIS
            mock_session.duration_seconds = 300
            mock_session.insights = [Mock(), Mock()]
            mock_session.patterns = [Mock()]
            mock_session.improvements = [Mock()]
            mock_session.discussion_summary = "Analysis completed"
            mock_session.action_items = ["Improve entry timing"]
            mock_analysis.return_value = mock_session
            
            session_summary = await phase4_integration.conduct_offhours_analysis_session(DiscussionTopic.TRADE_ANALYSIS)
            
            assert isinstance(session_summary, dict)
            assert session_summary["session_id"] == "test-session-123"
            assert session_summary["topic"] == DiscussionTopic.TRADE_ANALYSIS.value
            assert session_summary["insights_count"] == 2
            assert session_summary["patterns_count"] == 1
            assert session_summary["improvements_count"] == 1
    
    @pytest.mark.asyncio
    async def test_run_ensemble_health_check(self, phase4_integration):
        """Test running ensemble health check."""
        # Mock component status
        with patch.object(phase4_integration, 'get_phase4_status') as mock_status:
            mock_status.return_value = Mock(
                multi_model_integration=True,
                collaborative_discussion=True,
                offhours_discussion=True,
                phase3_integration=True,
                overall_status="FULLY_OPERATIONAL"
            )
            
            with patch.object(phase4_integration.phase3_integration, 'get_system_health') as mock_health:
                mock_health.return_value = {"overall_health": "HEALTHY"}
                
                with patch.object(phase4_integration, 'get_ensemble_metrics') as mock_metrics:
                    mock_metrics.return_value = Mock(
                        total_discussions=10,
                        successful_discussions=8,
                        average_confidence=0.75,
                        consensus_rate=0.8,
                        model_availability_rate=1.0,
                        offhours_sessions=5,
                        insights_generated=20,
                        improvements_suggested=5
                    )
                    
                    health_check = await phase4_integration.run_ensemble_health_check()
                    
                    assert isinstance(health_check, dict)
                    assert "phase4_status" in health_check
                    assert "system_health" in health_check
                    assert "ensemble_metrics" in health_check
                    assert "overall_health" in health_check
                    assert health_check["overall_health"] == "HEALTHY"

class TestPhase4Main:
    """Test Phase 4 Main."""
    
    @pytest.fixture
    def phase4_main(self):
        """Create Phase4Main instance."""
        return Phase4Main("DEMO")
    
    @pytest.mark.asyncio
    async def test_initialize(self, phase4_main):
        """Test Phase 4 main initialization."""
        # Mock Phase 4 integration
        with patch.object(phase4_main.phase4_integration, 'initialize_phase4') as mock_init:
            mock_init.return_value = True
            
            with patch.object(phase4_main.phase4_integration, 'validate_phase4_integration') as mock_validate:
                mock_validate.return_value = True
                
                result = await phase4_main.initialize()
                assert result == True
    
    @pytest.mark.asyncio
    async def test_start(self, phase4_main):
        """Test Phase 4 main start."""
        # Mock Phase 4 integration
        with patch.object(phase4_main.phase4_integration, 'start_phase4') as mock_start:
            mock_start.return_value = True
            
            result = await phase4_main.start()
            assert result == True
            assert phase4_main.is_running == True
    
    @pytest.mark.asyncio
    async def test_stop(self, phase4_main):
        """Test Phase 4 main stop."""
        # Mock Phase 4 integration
        with patch.object(phase4_main.phase4_integration, 'stop_phase4') as mock_stop:
            mock_stop.return_value = True
            
            result = await phase4_main.stop()
            assert result == True
            assert phase4_main.is_running == False
    
    @pytest.mark.asyncio
    async def test_run_health_check(self, phase4_main):
        """Test running health check."""
        # Mock health check
        with patch.object(phase4_main.phase4_integration, 'run_ensemble_health_check') as mock_health:
            mock_health.return_value = {"overall_health": "HEALTHY"}
            
            with patch.object(phase4_main.phase4_integration, 'get_phase4_status') as mock_status:
                mock_status.return_value = Mock(overall_status="FULLY_OPERATIONAL")
                
                with patch.object(phase4_main.phase4_integration, 'get_ensemble_metrics') as mock_metrics:
                    mock_metrics.return_value = Mock(
                        total_discussions=10,
                        successful_discussions=8,
                        average_confidence=0.75,
                        consensus_rate=0.8,
                        model_availability_rate=1.0,
                        offhours_sessions=5,
                        insights_generated=20,
                        improvements_suggested=5
                    )
                    
                    health_summary = await phase4_main.run_health_check()
                    
                    assert isinstance(health_summary, dict)
                    assert "timestamp" in health_summary
                    assert "mode" in health_summary
                    assert "status" in health_summary
                    assert "health_check" in health_summary
                    assert "metrics" in health_summary
                    assert "is_running" in health_summary

class TestIntegration:
    """Integration tests for Phase 4."""
    
    @pytest.mark.asyncio
    async def test_full_phase4_workflow(self):
        """Test full Phase 4 workflow."""
        # This test would require actual Ollama models to be running
        # For now, we'll mock the entire workflow
        
        with patch('src.ai.multi_model.OllamaClient') as mock_client:
            # Mock Ollama client
            mock_client.return_value.generate = AsyncMock(return_value="BUY: Strong bullish momentum")
            mock_client.return_value.chat = AsyncMock(return_value="Analysis complete")
            
            # Create Phase 4 integration
            phase4_integration = Phase4Integration("DEMO")
            
            # Initialize Phase 4
            with patch.object(phase4_integration.phase3_integration, 'validate_phase3_integration') as mock_phase3:
                mock_phase3.return_value = True
                
                result = await phase4_integration.initialize_phase4()
                assert result == True
            
            # Get status
            status = await phase4_integration.get_phase4_status()
            assert isinstance(status, Phase4Status)
            
            # Make collaborative decision
            market_data = {
                "symbol": "AAPL",
                "price": 150.0,
                "volume": 1000000,
                "sentiment": 0.7
            }
            
            with patch('src.ai.collaborative_discussion.conduct_discussion') as mock_discussion:
                mock_session = Mock()
                mock_session.final_decision = Mock()
                mock_session.final_decision.recommendation = "BUY"
                mock_session.final_decision.confidence = 0.8
                mock_session.final_decision.reasoning = "Strong bullish momentum"
                mock_session.final_decision.model_contributions = {"qwen3-coder:480b-cloud": 0.8}
                mock_session.final_decision.discussion_summary = "Models agree on buy"
                mock_session.final_decision.timestamp = datetime.now()
                mock_discussion.return_value = mock_session
                
                decision = await phase4_integration.make_collaborative_decision("BUY_OR_SELL", market_data)
                assert isinstance(decision, CollaborativeDecision)
                assert decision.recommendation == "BUY"
            
            # Conduct off-hours analysis
            with patch('src.ai.offhours_discussion.conduct_offhours_analysis') as mock_analysis:
                mock_session = Mock()
                mock_session.session_id = "test-session-123"
                mock_session.topic = DiscussionTopic.TRADE_ANALYSIS
                mock_session.duration_seconds = 300
                mock_session.insights = [Mock(), Mock()]
                mock_session.patterns = [Mock()]
                mock_session.improvements = [Mock()]
                mock_session.discussion_summary = "Analysis completed"
                mock_session.action_items = ["Improve entry timing"]
                mock_analysis.return_value = mock_session
                
                session_summary = await phase4_integration.conduct_offhours_analysis_session(DiscussionTopic.TRADE_ANALYSIS)
                assert isinstance(session_summary, dict)
                assert session_summary["insights_count"] == 2
            
            # Run health check
            with patch.object(phase4_integration.phase3_integration, 'get_system_health') as mock_health:
                mock_health.return_value = {"overall_health": "HEALTHY"}
                
                health_check = await phase4_integration.run_ensemble_health_check()
                assert isinstance(health_check, dict)
                assert "overall_health" in health_check

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
