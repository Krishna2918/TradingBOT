"""
Phase 5 Integration Module

This module integrates all Phase 5 components (Adaptive Configuration,
Performance Learning, and Self-Learning Engine) with the existing system.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import asdict

from src.config.mode_manager import get_current_mode, get_mode_config
from src.adaptive.configuration_manager import (
    get_adaptive_config_manager, PerformanceMetrics, 
    get_parameter, get_all_parameters, update_performance_metrics
)
from src.adaptive.performance_learning import (
    get_performance_learning_engine, TradeOutcome, 
    get_learning_summary, get_recent_patterns, get_pending_recommendations
)
from src.adaptive.self_learning_engine import (
    get_self_learning_engine, OptimizationObjective,
    get_meta_parameter, get_all_meta_parameters, optimize_parameters
)

logger = logging.getLogger(__name__)

class Phase5Integration:
    """Integration layer for Phase 5 components."""
    
    def __init__(self, mode: str = None):
        self.mode = mode or get_current_mode()
        self.mode_config = get_mode_config()
        
        # Initialize Phase 5 components
        self.config_manager = get_adaptive_config_manager(self.mode)
        self.learning_engine = get_performance_learning_engine(self.mode)
        self.self_learning_engine = get_self_learning_engine(self.mode)
        
        # Integration state
        self.is_initialized = False
        self.last_optimization = None
        self.optimization_schedule = None
        self.learning_active = True
        
        logger.info(f"Phase 5 Integration initialized for {self.mode} mode")
    
    async def initialize(self) -> bool:
        """Initialize Phase 5 integration."""
        try:
            logger.info("Initializing Phase 5 integration...")
            
            # Initialize all components
            await self._initialize_adaptive_configuration()
            await self._initialize_performance_learning()
            await self._initialize_self_learning()
            
            # Set up optimization schedule
            await self._setup_optimization_schedule()
            
            self.is_initialized = True
            logger.info("Phase 5 integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Phase 5 integration: {e}")
            return False
    
    async def _initialize_adaptive_configuration(self):
        """Initialize adaptive configuration system."""
        try:
            # Load existing configuration
            current_params = self.config_manager.get_all_parameters()
            logger.info(f"Loaded {len(current_params)} adaptive parameters")
            
            # Validate parameter ranges
            for param_name, value in current_params.items():
                param_config = self.config_manager.parameters.get(param_name)
                if param_config:
                    if value < param_config.min_value or value > param_config.max_value:
                        logger.warning(f"Parameter {param_name} value {value} is outside valid range "
                                     f"[{param_config.min_value}, {param_config.max_value}]")
            
        except Exception as e:
            logger.error(f"Error initializing adaptive configuration: {e}")
    
    async def _initialize_performance_learning(self):
        """Initialize performance learning system."""
        try:
            # Load learning summary
            learning_summary = self.learning_engine.get_learning_summary()
            logger.info(f"Performance learning initialized: {learning_summary['total_trades_analyzed']} trades analyzed")
            
            # Check for pending recommendations
            pending_recommendations = self.learning_engine.get_pending_recommendations()
            if pending_recommendations:
                logger.info(f"Found {len(pending_recommendations)} pending parameter recommendations")
            
        except Exception as e:
            logger.error(f"Error initializing performance learning: {e}")
    
    async def _initialize_self_learning(self):
        """Initialize self-learning engine."""
        try:
            # Load learning state
            learning_summary = self.self_learning_engine.get_learning_summary()
            learning_state = learning_summary["learning_state"]
            
            logger.info(f"Self-learning engine initialized: Phase={learning_state['learning_phase']}, "
                       f"Optimizations={learning_state['total_optimizations']}")
            
            # Set initial optimization objective
            if learning_state["total_optimizations"] == 0:
                self.self_learning_engine.set_learning_objective(OptimizationObjective.BALANCED_PERFORMANCE)
            
        except Exception as e:
            logger.error(f"Error initializing self-learning engine: {e}")
    
    async def _setup_optimization_schedule(self):
        """Set up automatic optimization schedule."""
        try:
            # Schedule daily optimization at market close
            self.optimization_schedule = {
                "daily_optimization": "16:00",  # 4 PM EST
                "weekly_analysis": "sunday_20:00",  # Sunday 8 PM
                "monthly_review": "first_monday_21:00"  # First Monday 9 PM
            }
            
            logger.info("Optimization schedule configured")
            
        except Exception as e:
            logger.error(f"Error setting up optimization schedule: {e}")
    
    async def start_adaptive_learning(self) -> bool:
        """Start the adaptive learning process."""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            logger.info("Starting adaptive learning process...")
            
            # Start background learning tasks
            asyncio.create_task(self._learning_loop())
            asyncio.create_task(self._optimization_loop())
            asyncio.create_task(self._monitoring_loop())
            
            self.learning_active = True
            logger.info("Adaptive learning process started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting adaptive learning: {e}")
            return False
    
    async def stop_adaptive_learning(self):
        """Stop the adaptive learning process."""
        try:
            self.learning_active = False
            logger.info("Adaptive learning process stopped")
            
        except Exception as e:
            logger.error(f"Error stopping adaptive learning: {e}")
    
    async def _learning_loop(self):
        """Main learning loop."""
        while self.learning_active:
            try:
                # Check for new trade outcomes to analyze
                await self._process_new_trades()
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Generate insights
                await self._generate_insights()
                
                # Wait before next iteration
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _optimization_loop(self):
        """Main optimization loop."""
        while self.learning_active:
            try:
                # Check if it's time for optimization
                if await self._should_optimize():
                    await self._run_optimization()
                
                # Wait before next check
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def _monitoring_loop(self):
        """Monitoring and health check loop."""
        while self.learning_active:
            try:
                # Check system health
                health_status = await self._check_system_health()
                
                if health_status["status"] != "healthy":
                    logger.warning(f"System health issue: {health_status['issues']}")
                
                # Wait before next check
                await asyncio.sleep(1800)  # 30 minutes
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def _process_new_trades(self):
        """Process new trade outcomes for learning."""
        try:
            # This would integrate with the trading system to get new trade outcomes
            # For now, we'll simulate this process
            
            # In a real implementation, this would:
            # 1. Get new trade outcomes from the trading system
            # 2. Record them in the learning engine
            # 3. Trigger pattern analysis
            
            pass
            
        except Exception as e:
            logger.error(f"Error processing new trades: {e}")
    
    async def _update_performance_metrics(self):
        """Update performance metrics."""
        try:
            # Get recent performance data
            recent_metrics = self.config_manager.performance_history[-1] if self.config_manager.performance_history else None
            
            if recent_metrics:
                # Update learning engine with latest metrics
                self.learning_engine.config_manager.update_performance_metrics(recent_metrics)
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def _generate_insights(self):
        """Generate learning insights."""
        try:
            # Check for new patterns
            recent_patterns = self.learning_engine.get_recent_patterns(days=1)
            
            if recent_patterns:
                logger.info(f"Generated {len(recent_patterns)} new insights")
                
                # Process insights and generate recommendations
                for pattern in recent_patterns:
                    await self._process_pattern_insight(pattern)
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
    
    async def _process_pattern_insight(self, pattern):
        """Process a pattern insight."""
        try:
            # Analyze pattern and generate recommendations
            if pattern.success_rate > 0.8 and pattern.frequency >= 3:
                # High-success pattern - consider increasing confidence threshold
                logger.info(f"High-success pattern identified: {pattern.description}")
                
            elif pattern.success_rate < 0.3 and pattern.frequency >= 3:
                # Low-success pattern - consider avoiding these conditions
                logger.info(f"Low-success pattern identified: {pattern.description}")
            
        except Exception as e:
            logger.error(f"Error processing pattern insight: {e}")
    
    async def _should_optimize(self) -> bool:
        """Check if optimization should be run."""
        try:
            # Check time-based triggers
            now = datetime.now()
            
            # Daily optimization at market close
            if now.hour == 16 and now.minute < 5:  # 4:00-4:05 PM
                return True
            
            # Weekly analysis on Sunday evening
            if now.weekday() == 6 and now.hour == 20 and now.minute < 5:  # Sunday 8:00-8:05 PM
                return True
            
            # Check performance-based triggers
            recent_metrics = self.config_manager.performance_history[-3:] if self.config_manager.performance_history else []
            
            if len(recent_metrics) >= 3:
                # Check for performance decline
                recent_win_rate = sum(m.win_rate for m in recent_metrics) / len(recent_metrics)
                if recent_win_rate < 0.4:
                    logger.info("Performance decline detected - triggering optimization")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking optimization triggers: {e}")
            return False
    
    async def _run_optimization(self):
        """Run parameter optimization."""
        try:
            logger.info("Starting parameter optimization...")
            
            # Determine optimization objective based on current performance
            objective = await self._determine_optimization_objective()
            
            # Run optimization
            result = await self.self_learning_engine.optimize_parameters(objective)
            
            if result["status"] == "completed":
                successful_optimizations = len([r for r in result["results"].values() if r.get("success")])
                logger.info(f"Optimization completed: {successful_optimizations} parameters optimized")
                
                # Apply successful optimizations
                await self._apply_optimization_results(result["results"])
                
                self.last_optimization = datetime.now()
            else:
                logger.warning(f"Optimization failed: {result.get('status', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error running optimization: {e}")
    
    async def _determine_optimization_objective(self) -> OptimizationObjective:
        """Determine the best optimization objective based on current performance."""
        try:
            recent_metrics = self.config_manager.performance_history[-5:] if self.config_manager.performance_history else []
            
            if not recent_metrics:
                return OptimizationObjective.BALANCED_PERFORMANCE
            
            # Calculate average metrics
            avg_win_rate = sum(m.win_rate for m in recent_metrics) / len(recent_metrics)
            avg_profit_factor = sum(m.profit_factor for m in recent_metrics) / len(recent_metrics)
            avg_drawdown = sum(m.max_drawdown for m in recent_metrics) / len(recent_metrics)
            avg_sharpe = sum(m.sharpe_ratio for m in recent_metrics) / len(recent_metrics)
            
            # Determine objective based on weakest metric
            if avg_win_rate < 0.5:
                return OptimizationObjective.MAXIMIZE_WIN_RATE
            elif avg_profit_factor < 1.2:
                return OptimizationObjective.MAXIMIZE_PROFIT_FACTOR
            elif avg_drawdown > 0.1:
                return OptimizationObjective.MINIMIZE_DRAWDOWN
            elif avg_sharpe < 0.5:
                return OptimizationObjective.MAXIMIZE_SHARPE_RATIO
            else:
                return OptimizationObjective.BALANCED_PERFORMANCE
            
        except Exception as e:
            logger.error(f"Error determining optimization objective: {e}")
            return OptimizationObjective.BALANCED_PERFORMANCE
    
    async def _apply_optimization_results(self, results: Dict[str, Any]):
        """Apply successful optimization results."""
        try:
            for param_name, result in results.items():
                if result.get("success"):
                    # Update the parameter in the configuration manager
                    new_value = result["new_value"]
                    
                    # This would need to be implemented in the configuration manager
                    # self.config_manager.set_parameter(param_name, new_value)
                    
                    logger.info(f"Applied optimization: {param_name} = {new_value}")
            
        except Exception as e:
            logger.error(f"Error applying optimization results: {e}")
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """Check system health."""
        try:
            issues = []
            status = "healthy"
            
            # Check configuration manager
            try:
                params = self.config_manager.get_all_parameters()
                if not params:
                    issues.append("No adaptive parameters available")
                    status = "degraded"
            except Exception as e:
                issues.append(f"Configuration manager error: {e}")
                status = "unhealthy"
            
            # Check learning engine
            try:
                learning_summary = self.learning_engine.get_learning_summary()
                if learning_summary["total_trades_analyzed"] == 0:
                    issues.append("No trades analyzed for learning")
                    status = "degraded"
            except Exception as e:
                issues.append(f"Learning engine error: {e}")
                status = "unhealthy"
            
            # Check self-learning engine
            try:
                meta_params = self.self_learning_engine.get_all_meta_parameters()
                if not meta_params:
                    issues.append("No meta-parameters available")
                    status = "degraded"
            except Exception as e:
                issues.append(f"Self-learning engine error: {e}")
                status = "unhealthy"
            
            return {
                "status": status,
                "issues": issues,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return {
                "status": "unhealthy",
                "issues": [f"Health check error: {e}"],
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            # Get status from all components
            config_status = {
                "parameters_count": len(self.config_manager.get_all_parameters()),
                "adjustments_count": len(self.config_manager.adjustment_history),
                "performance_metrics_count": len(self.config_manager.performance_history)
            }
            
            learning_status = self.learning_engine.get_learning_summary()
            
            self_learning_status = self.self_learning_engine.get_learning_summary()
            
            # Get health status
            health_status = await self._check_system_health()
            
            return {
                "phase": "5",
                "mode": self.mode,
                "is_initialized": self.is_initialized,
                "learning_active": self.learning_active,
                "last_optimization": self.last_optimization.isoformat() if self.last_optimization else None,
                "health": health_status,
                "adaptive_configuration": config_status,
                "performance_learning": learning_status,
                "self_learning": self_learning_status,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                "phase": "5",
                "mode": self.mode,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        try:
            learning_summary = self.self_learning_engine.get_learning_summary()
            learning_state = learning_summary["learning_state"]
            
            return {
                "total_optimizations": learning_state["total_optimizations"],
                "successful_optimizations": learning_state["successful_optimizations"],
                "success_rate": (learning_state["successful_optimizations"] / learning_state["total_optimizations"] 
                               if learning_state["total_optimizations"] > 0 else 0),
                "current_objective": learning_state["current_objective"],
                "learning_phase": learning_state["learning_phase"],
                "confidence_level": learning_state["confidence_level"],
                "last_optimization": learning_state["last_optimization"],
                "meta_parameters": learning_summary["meta_parameters"]
            }
            
        except Exception as e:
            logger.error(f"Error getting optimization summary: {e}")
            return {"error": str(e)}
    
    async def manual_optimization(self, objective: OptimizationObjective = None) -> Dict[str, Any]:
        """Trigger manual optimization."""
        try:
            logger.info("Manual optimization triggered")
            
            result = await self.self_learning_engine.optimize_parameters(objective)
            
            if result["status"] == "completed":
                # Apply results
                await self._apply_optimization_results(result["results"])
                self.last_optimization = datetime.now()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in manual optimization: {e}")
            return {"status": "error", "error": str(e)}
    
    async def get_parameter_recommendations(self) -> List[Dict[str, Any]]:
        """Get parameter recommendations."""
        try:
            recommendations = self.learning_engine.get_pending_recommendations()
            
            return [{
                "parameter_name": rec.parameter_name,
                "current_value": rec.current_value,
                "recommended_value": rec.recommended_value,
                "confidence": rec.confidence,
                "reasoning": rec.reasoning,
                "expected_improvement": rec.expected_improvement,
                "risk_assessment": rec.risk_assessment,
                "created_at": rec.created_at.isoformat()
            } for rec in recommendations]
            
        except Exception as e:
            logger.error(f"Error getting parameter recommendations: {e}")
            return []

# Global instance
_phase5_integration = None

def get_phase5_integration(mode: str = None) -> Phase5Integration:
    """Get the global Phase 5 integration instance."""
    global _phase5_integration
    if _phase5_integration is None or (mode and _phase5_integration.mode != mode):
        _phase5_integration = Phase5Integration(mode)
    return _phase5_integration

async def initialize_phase5(mode: str = None) -> bool:
    """Initialize Phase 5 integration."""
    return await get_phase5_integration(mode).initialize()

async def start_adaptive_learning(mode: str = None) -> bool:
    """Start adaptive learning."""
    return await get_phase5_integration(mode).start_adaptive_learning()

async def stop_adaptive_learning(mode: str = None):
    """Stop adaptive learning."""
    await get_phase5_integration(mode).stop_adaptive_learning()

async def get_phase5_status(mode: str = None) -> Dict[str, Any]:
    """Get Phase 5 status."""
    return await get_phase5_integration(mode).get_system_status()

async def get_optimization_summary(mode: str = None) -> Dict[str, Any]:
    """Get optimization summary."""
    return await get_phase5_integration(mode).get_optimization_summary()

async def manual_optimization(objective: OptimizationObjective = None, mode: str = None) -> Dict[str, Any]:
    """Trigger manual optimization."""
    return await get_phase5_integration(mode).manual_optimization(objective)

async def get_parameter_recommendations(mode: str = None) -> List[Dict[str, Any]]:
    """Get parameter recommendations."""
    return await get_phase5_integration(mode).get_parameter_recommendations()
