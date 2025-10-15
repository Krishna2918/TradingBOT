"""
Phase 4 Integration Module

This module integrates all Phase 4 components with the existing system,
providing a unified interface for the collaborative AI ensemble system.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from src.ai.multi_model import get_multi_model_manager, check_model_availability, get_all_model_opinions
from src.ai.collaborative_discussion import get_collaborative_discussion, conduct_discussion, get_discussion_statistics
from src.ai.offhours_discussion import get_offhours_ai, conduct_offhours_analysis, DiscussionTopic, get_insights_summary
from src.workflows.phase3_integration import get_phase3_integration, get_system_health
from src.config.mode_manager import get_current_mode

logger = logging.getLogger(__name__)

@dataclass
class Phase4Status:
    """Status of Phase 4 components."""
    multi_model_integration: bool
    collaborative_discussion: bool
    offhours_discussion: bool
    model_availability: Dict[str, bool]
    phase3_integration: bool
    overall_status: str

@dataclass
class CollaborativeDecision:
    """Collaborative decision from AI ensemble."""
    topic: str
    recommendation: str
    confidence: float
    reasoning: str
    model_contributions: Dict[str, float]
    discussion_summary: str
    timestamp: datetime

@dataclass
class AIEnsembleMetrics:
    """Metrics for the AI ensemble system."""
    total_discussions: int
    successful_discussions: int
    average_confidence: float
    consensus_rate: float
    model_availability_rate: float
    offhours_sessions: int
    insights_generated: int
    improvements_suggested: int

class Phase4Integration:
    """Integrates all Phase 4 components with the existing system."""
    
    def __init__(self, mode: str = "DEMO"):
        self.mode = mode
        
        # Initialize Phase 4 components
        self.multi_model_manager = get_multi_model_manager(mode)
        self.collaborative_discussion = get_collaborative_discussion(mode)
        self.offhours_ai = get_offhours_ai(mode)
        
        # Initialize Phase 3 integration
        self.phase3_integration = get_phase3_integration(mode)
        
        # Integration state
        self.is_initialized = False
        self.is_running = False
        
        # Performance tracking
        self.collaborative_decisions = []
        self.ensemble_metrics = AIEnsembleMetrics(
            total_discussions=0,
            successful_discussions=0,
            average_confidence=0.0,
            consensus_rate=0.0,
            model_availability_rate=0.0,
            offhours_sessions=0,
            insights_generated=0,
            improvements_suggested=0
        )
        
        logger.info(f"Phase 4 Integration initialized for {mode} mode")
    
    async def initialize_phase4(self) -> bool:
        """Initialize all Phase 4 components."""
        try:
            logger.info("Initializing Phase 4 components...")
            
            # Check model availability
            model_availability = await self.multi_model_manager.check_model_availability()
            available_models = sum(1 for available in model_availability.values() if available)
            total_models = len(model_availability)
            
            if available_models == 0:
                logger.error("No AI models are available")
                return False
            
            if available_models < total_models:
                logger.warning(f"Only {available_models}/{total_models} models are available")
            
            # Validate collaborative discussion system
            if not self.collaborative_discussion.validate_discussion():
                logger.error("Failed to initialize Collaborative Discussion")
                return False
            
            # Validate off-hours discussion system
            if not self.offhours_ai.validate_offhours_system():
                logger.error("Failed to initialize Off-Hours Discussion")
                return False
            
            # Validate Phase 3 integration
            if not self.phase3_integration.validate_phase3_integration():
                logger.error("Failed to initialize Phase 3 Integration")
                return False
            
            self.is_initialized = True
            logger.info("Phase 4 components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Phase 4: {e}")
            return False
    
    async def start_phase4(self) -> bool:
        """Start Phase 4 operations."""
        if not self.is_initialized:
            logger.error("Phase 4 not initialized. Call initialize_phase4() first.")
            return False
        
        try:
            logger.info("Starting Phase 4 operations...")
            
            # Start Phase 3 operations first
            if not self.phase3_integration.start_phase3():
                logger.warning("Failed to start Phase 3 operations")
            
            # Initialize ensemble metrics
            await self._update_ensemble_metrics()
            
            self.is_running = True
            logger.info("Phase 4 operations started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting Phase 4: {e}")
            return False
    
    async def stop_phase4(self) -> bool:
        """Stop Phase 4 operations."""
        try:
            logger.info("Stopping Phase 4 operations...")
            
            # Stop Phase 3 operations
            if not self.phase3_integration.stop_phase3():
                logger.warning("Failed to stop Phase 3 operations")
            
            self.is_running = False
            logger.info("Phase 4 operations stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping Phase 4: {e}")
            return False
    
    async def get_phase4_status(self) -> Phase4Status:
        """Get status of all Phase 4 components."""
        try:
            # Check model availability
            model_availability = await self.multi_model_manager.check_model_availability()
            multi_model_operational = any(model_availability.values())
            
            # Check collaborative discussion
            collaborative_operational = self.collaborative_discussion.validate_discussion()
            
            # Check off-hours discussion
            offhours_operational = self.offhours_ai.validate_offhours_system()
            
            # Check Phase 3 integration
            phase3_status = self.phase3_integration.get_phase3_status()
            phase3_operational = phase3_status.overall_status == "FULLY_OPERATIONAL"
            
            # Determine overall status
            if (multi_model_operational and collaborative_operational and 
                offhours_operational and phase3_operational):
                overall_status = "FULLY_OPERATIONAL"
            elif (multi_model_operational or collaborative_operational or 
                  offhours_operational or phase3_operational):
                overall_status = "PARTIALLY_OPERATIONAL"
            else:
                overall_status = "NOT_OPERATIONAL"
            
            return Phase4Status(
                multi_model_integration=multi_model_operational,
                collaborative_discussion=collaborative_operational,
                offhours_discussion=offhours_operational,
                model_availability=model_availability,
                phase3_integration=phase3_operational,
                overall_status=overall_status
            )
            
        except Exception as e:
            logger.error(f"Error getting Phase 4 status: {e}")
            return Phase4Status(
                multi_model_integration=False,
                collaborative_discussion=False,
                offhours_discussion=False,
                model_availability={},
                phase3_integration=False,
                overall_status="ERROR"
            )
    
    async def make_collaborative_decision(self, topic: str, market_data: Dict[str, Any]) -> Optional[CollaborativeDecision]:
        """Make a collaborative decision using the AI ensemble."""
        try:
            logger.info(f"Making collaborative decision: {topic}")
            
            # Conduct collaborative discussion
            discussion_session = await conduct_discussion(topic, market_data, self.mode)
            
            if not discussion_session.final_decision:
                logger.error("No final decision reached in discussion")
                return None
            
            final_decision = discussion_session.final_decision
            
            # Create collaborative decision
            collaborative_decision = CollaborativeDecision(
                topic=topic,
                recommendation=final_decision.recommendation,
                confidence=final_decision.confidence,
                reasoning=final_decision.reasoning,
                model_contributions=final_decision.model_contributions,
                discussion_summary=final_decision.discussion_summary,
                timestamp=final_decision.timestamp
            )
            
            # Store decision
            self.collaborative_decisions.append(collaborative_decision)
            
            # Update metrics
            self.ensemble_metrics.total_discussions += 1
            if final_decision.confidence > 0.6:
                self.ensemble_metrics.successful_discussions += 1
            
            logger.info(f"Collaborative decision made: {final_decision.recommendation} (confidence: {final_decision.confidence:.2f})")
            return collaborative_decision
            
        except Exception as e:
            logger.error(f"Error making collaborative decision: {e}")
            return None
    
    async def conduct_offhours_analysis_session(self, topic: DiscussionTopic) -> Optional[Dict[str, Any]]:
        """Conduct an off-hours analysis session."""
        try:
            logger.info(f"Conducting off-hours analysis: {topic.value}")
            
            # Conduct off-hours analysis
            offhours_session = await conduct_offhours_analysis(topic, self.mode)
            
            # Update metrics
            self.ensemble_metrics.offhours_sessions += 1
            self.ensemble_metrics.insights_generated += len(offhours_session.insights)
            self.ensemble_metrics.improvements_suggested += len(offhours_session.improvements)
            
            # Create session summary
            session_summary = {
                "session_id": offhours_session.session_id,
                "topic": offhours_session.topic.value,
                "duration_seconds": offhours_session.duration_seconds,
                "insights_count": len(offhours_session.insights),
                "patterns_count": len(offhours_session.patterns),
                "improvements_count": len(offhours_session.improvements),
                "discussion_summary": offhours_session.discussion_summary,
                "action_items": offhours_session.action_items
            }
            
            logger.info(f"Off-hours analysis completed: {len(offhours_session.insights)} insights, {len(offhours_session.improvements)} improvements")
            return session_summary
            
        except Exception as e:
            logger.error(f"Error conducting off-hours analysis: {e}")
            return None
    
    async def get_ensemble_metrics(self) -> AIEnsembleMetrics:
        """Get comprehensive AI ensemble metrics."""
        try:
            # Update metrics
            await self._update_ensemble_metrics()
            
            return self.ensemble_metrics
            
        except Exception as e:
            logger.error(f"Error getting ensemble metrics: {e}")
            return self.ensemble_metrics
    
    async def _update_ensemble_metrics(self):
        """Update ensemble metrics."""
        try:
            # Update discussion statistics
            discussion_stats = get_discussion_statistics(self.mode)
            self.ensemble_metrics.total_discussions = discussion_stats.get("total_sessions", 0)
            self.ensemble_metrics.successful_discussions = discussion_stats.get("successful_sessions", 0)
            
            # Calculate consensus rate
            if self.ensemble_metrics.total_discussions > 0:
                self.ensemble_metrics.consensus_rate = (
                    self.ensemble_metrics.successful_discussions / self.ensemble_metrics.total_discussions
                )
            
            # Calculate average confidence
            if self.collaborative_decisions:
                total_confidence = sum(decision.confidence for decision in self.collaborative_decisions)
                self.ensemble_metrics.average_confidence = total_confidence / len(self.collaborative_decisions)
            
            # Update model availability rate
            model_availability = await self.multi_model_manager.check_model_availability()
            available_models = sum(1 for available in model_availability.values() if available)
            total_models = len(model_availability)
            self.ensemble_metrics.model_availability_rate = available_models / total_models if total_models > 0 else 0
            
            # Update insights and improvements from off-hours AI
            insights_summary = get_insights_summary(self.mode)
            self.ensemble_metrics.insights_generated = insights_summary.get("total_insights", 0)
            self.ensemble_metrics.improvements_suggested = insights_summary.get("total_improvements", 0)
            
        except Exception as e:
            logger.error(f"Error updating ensemble metrics: {e}")
    
    async def run_ensemble_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check for the AI ensemble."""
        try:
            logger.info("Running AI ensemble health check...")
            
            # Get Phase 4 status
            phase4_status = await self.get_phase4_status()
            
            # Get system health from Phase 3
            system_health = self.phase3_integration.get_system_health()
            
            # Get ensemble metrics
            ensemble_metrics = await self.get_ensemble_metrics()
            
            # Check individual components
            health_checks = {
                "phase4_status": phase4_status,
                "system_health": system_health,
                "ensemble_metrics": ensemble_metrics,
                "model_health": await self._check_model_health(),
                "discussion_health": self._check_discussion_health(),
                "offhours_health": self._check_offhours_health()
            }
            
            # Determine overall health
            overall_health = "HEALTHY"
            if phase4_status.overall_status != "FULLY_OPERATIONAL":
                overall_health = "DEGRADED"
            if ensemble_metrics.model_availability_rate < 0.5:
                overall_health = "UNHEALTHY"
            
            health_checks["overall_health"] = overall_health
            
            logger.info(f"AI ensemble health check completed: {overall_health}")
            return health_checks
            
        except Exception as e:
            logger.error(f"Error in ensemble health check: {e}")
            return {"error": str(e), "overall_health": "ERROR"}
    
    async def _check_model_health(self) -> Dict[str, Any]:
        """Check health of AI models."""
        try:
            model_availability = await self.multi_model_manager.check_model_availability()
            available_models = sum(1 for available in model_availability.values() if available)
            total_models = len(model_availability)
            
            health_status = "HEALTHY"
            issues = []
            
            if available_models < total_models:
                health_status = "DEGRADED"
                unavailable_models = [name for name, available in model_availability.items() if not available]
                issues.append(f"Unavailable models: {', '.join(unavailable_models)}")
            
            if available_models == 0:
                health_status = "UNHEALTHY"
                issues.append("No models available")
            
            return {
                "status": health_status,
                "issues": issues,
                "available_models": available_models,
                "total_models": total_models,
                "availability_rate": available_models / total_models if total_models > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error checking model health: {e}")
            return {"status": "ERROR", "issues": [str(e)], "available_models": 0, "total_models": 0, "availability_rate": 0}
    
    def _check_discussion_health(self) -> Dict[str, Any]:
        """Check health of collaborative discussion system."""
        try:
            discussion_stats = get_discussion_statistics(self.mode)
            
            health_status = "HEALTHY"
            issues = []
            
            success_rate = discussion_stats.get("success_rate", 0)
            if success_rate < 0.8:
                health_status = "DEGRADED"
                issues.append(f"Low discussion success rate: {success_rate:.1%}")
            
            avg_duration = discussion_stats.get("average_duration_seconds", 0)
            if avg_duration > 300:  # 5 minutes
                health_status = "DEGRADED"
                issues.append(f"Long discussion duration: {avg_duration:.1f}s")
            
            return {
                "status": health_status,
                "issues": issues,
                "success_rate": success_rate,
                "average_duration": avg_duration,
                "total_sessions": discussion_stats.get("total_sessions", 0)
            }
            
        except Exception as e:
            logger.error(f"Error checking discussion health: {e}")
            return {"status": "ERROR", "issues": [str(e)], "success_rate": 0, "average_duration": 0, "total_sessions": 0}
    
    def _check_offhours_health(self) -> Dict[str, Any]:
        """Check health of off-hours discussion system."""
        try:
            insights_summary = get_insights_summary(self.mode)
            
            health_status = "HEALTHY"
            issues = []
            
            total_insights = insights_summary.get("total_insights", 0)
            if total_insights == 0:
                health_status = "DEGRADED"
                issues.append("No insights generated")
            
            return {
                "status": health_status,
                "issues": issues,
                "total_insights": total_insights,
                "success_insights": insights_summary.get("success_insights", 0),
                "failure_insights": insights_summary.get("failure_insights", 0),
                "total_improvements": insights_summary.get("total_improvements", 0)
            }
            
        except Exception as e:
            logger.error(f"Error checking off-hours health: {e}")
            return {"status": "ERROR", "issues": [str(e)], "total_insights": 0, "success_insights": 0, "failure_insights": 0, "total_improvements": 0}
    
    def get_collaborative_decisions_history(self, limit: int = 10) -> List[CollaborativeDecision]:
        """Get recent collaborative decisions."""
        return self.collaborative_decisions[-limit:] if self.collaborative_decisions else []
    
    async def validate_phase4_integration(self) -> bool:
        """Validate Phase 4 integration."""
        try:
            # Check if all components are initialized
            if not self.is_initialized:
                logger.error("Phase 4 not initialized")
                return False
            
            # Check component status
            status = await self.get_phase4_status()
            
            if status.overall_status == "NOT_OPERATIONAL":
                logger.error("Phase 4 components not operational")
                return False
            
            # Check if at least one model is available
            if not any(status.model_availability.values()):
                logger.error("No AI models are available")
                return False
            
            # Check system health
            health_check = await self.run_ensemble_health_check()
            
            if health_check.get("overall_health") == "UNHEALTHY":
                logger.error("AI ensemble health is unhealthy")
                return False
            
            logger.info("Phase 4 integration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Phase 4 integration validation error: {e}")
            return False

# Global Phase 4 integration instance
_phase4_integration: Optional[Phase4Integration] = None

def get_phase4_integration(mode: str = None) -> Phase4Integration:
    """Get the global Phase 4 integration instance."""
    global _phase4_integration
    if _phase4_integration is None:
        if mode is None:
            mode = get_current_mode()
        _phase4_integration = Phase4Integration(mode)
    return _phase4_integration

async def initialize_phase4(mode: str = None) -> bool:
    """Initialize Phase 4 components."""
    return await get_phase4_integration(mode).initialize_phase4()

async def start_phase4(mode: str = None) -> bool:
    """Start Phase 4 operations."""
    return await get_phase4_integration(mode).start_phase4()

async def stop_phase4(mode: str = None) -> bool:
    """Stop Phase 4 operations."""
    return await get_phase4_integration(mode).stop_phase4()

async def get_phase4_status(mode: str = None) -> Phase4Status:
    """Get Phase 4 status."""
    return await get_phase4_integration(mode).get_phase4_status()

async def make_collaborative_decision(topic: str, market_data: Dict[str, Any], mode: str = None) -> Optional[CollaborativeDecision]:
    """Make a collaborative decision."""
    return await get_phase4_integration(mode).make_collaborative_decision(topic, market_data)

async def conduct_offhours_analysis_session(topic: DiscussionTopic, mode: str = None) -> Optional[Dict[str, Any]]:
    """Conduct off-hours analysis session."""
    return await get_phase4_integration(mode).conduct_offhours_analysis_session(topic)

async def get_ensemble_metrics(mode: str = None) -> AIEnsembleMetrics:
    """Get ensemble metrics."""
    return await get_phase4_integration(mode).get_ensemble_metrics()

async def run_ensemble_health_check(mode: str = None) -> Dict[str, Any]:
    """Run ensemble health check."""
    return await get_phase4_integration(mode).run_ensemble_health_check()

def get_collaborative_decisions_history(limit: int = 10, mode: str = None) -> List[CollaborativeDecision]:
    """Get collaborative decisions history."""
    return get_phase4_integration(mode).get_collaborative_decisions_history(limit)

async def validate_phase4_integration(mode: str = None) -> bool:
    """Validate Phase 4 integration."""
    return await get_phase4_integration(mode).validate_phase4_integration()
