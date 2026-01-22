"""
Phase 2 Integration Module

This module integrates all Phase 2 components with Phase 1, providing
a unified interface for the enhanced AI trading system.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from src.ai.enhanced_ensemble import get_enhanced_ensemble, analyze_for_entry, analyze_for_exit
from src.workflows.trading_cycle import get_trading_cycle, execute_complete_cycle
from src.workflows.activity_scheduler import get_activity_scheduler, start_scheduler, stop_scheduler, get_scheduler_status
from src.trading.positions import get_position_manager, get_open_positions, get_portfolio_summary
from src.trading.risk import get_risk_summary
from src.config.mode_manager import get_current_mode

logger = logging.getLogger(__name__)

@dataclass
class Phase2Status:
    """Status of Phase 2 components."""
    enhanced_ensemble: bool
    trading_cycle: bool
    activity_scheduler: bool
    integration: bool
    overall_status: str

@dataclass
class SystemMetrics:
    """System performance metrics."""
    total_positions: int
    open_positions: int
    closed_positions: int
    total_pnl: float
    win_rate: float
    risk_status: str
    ai_confidence: float
    system_health: str

class Phase2Integration:
    """Integrates all Phase 2 components with Phase 1."""
    
    def __init__(self):
        """Initialize Phase 2 Integration."""
        self.enhanced_ensemble = get_enhanced_ensemble()
        self.trading_cycle = get_trading_cycle()
        self.activity_scheduler = get_activity_scheduler()
        
        # Integration state
        self.is_initialized = False
        self.is_running = False
        
        logger.info("Phase 2 Integration initialized")
    
    def initialize_phase2(self) -> bool:
        """Initialize all Phase 2 components."""
        try:
            logger.info("Initializing Phase 2 components...")
            
            # Initialize Enhanced AI Ensemble
            if not self.enhanced_ensemble.initialize_ensemble():
                logger.error("Failed to initialize Enhanced AI Ensemble")
                return False
            
            # Initialize Trading Cycle
            if not self.trading_cycle.validate_cycle():
                logger.error("Failed to initialize Trading Cycle")
                return False
            
            # Initialize Activity Scheduler
            # Note: Scheduler is initialized but not started yet
            
            self.is_initialized = True
            logger.info("Phase 2 components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Phase 2: {e}")
            return False
    
    def start_phase2(self) -> bool:
        """Start Phase 2 operations."""
        if not self.is_initialized:
            logger.error("Phase 2 not initialized. Call initialize_phase2() first.")
            return False
        
        try:
            logger.info("Starting Phase 2 operations...")
            
            # Start Activity Scheduler
            start_scheduler()
            
            self.is_running = True
            logger.info("Phase 2 operations started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting Phase 2: {e}")
            return False
    
    def stop_phase2(self) -> bool:
        """Stop Phase 2 operations."""
        try:
            logger.info("Stopping Phase 2 operations...")
            
            # Stop Activity Scheduler
            stop_scheduler()
            
            self.is_running = False
            logger.info("Phase 2 operations stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping Phase 2: {e}")
            return False
    
    def get_phase2_status(self) -> Phase2Status:
        """Get status of all Phase 2 components."""
        try:
            # Check Enhanced AI Ensemble
            ensemble_status = self.enhanced_ensemble.validate_ensemble()
            
            # Check Trading Cycle
            cycle_status = self.trading_cycle.validate_cycle()
            
            # Check Activity Scheduler
            scheduler_status = get_scheduler_status()
            scheduler_running = scheduler_status.get("is_running", False)
            
            # Check Integration
            integration_status = self.is_initialized and self.is_running
            
            # Determine overall status
            if ensemble_status and cycle_status and scheduler_running and integration_status:
                overall_status = "FULLY_OPERATIONAL"
            elif ensemble_status and cycle_status and integration_status:
                overall_status = "PARTIALLY_OPERATIONAL"
            else:
                overall_status = "NOT_OPERATIONAL"
            
            return Phase2Status(
                enhanced_ensemble=ensemble_status,
                trading_cycle=cycle_status,
                activity_scheduler=scheduler_running,
                integration=integration_status,
                overall_status=overall_status
            )
            
        except Exception as e:
            logger.error(f"Error getting Phase 2 status: {e}")
            return Phase2Status(
                enhanced_ensemble=False,
                trading_cycle=False,
                activity_scheduler=False,
                integration=False,
                overall_status="ERROR"
            )
    
    def get_system_metrics(self, mode: Optional[str] = None) -> SystemMetrics:
        """Get comprehensive system metrics."""
        if mode is None:
            mode = get_current_mode()
        
        try:
            # Get portfolio summary
            portfolio_summary = get_portfolio_summary(mode)
            
            # Get risk summary
            risk_summary = get_risk_summary(mode)
            
            # Get AI confidence (average from recent analyses)
            ai_confidence = self._get_average_ai_confidence()
            
            # Get system health
            system_health = self._get_system_health()
            
            # Calculate win rate
            win_rate = self._calculate_win_rate(mode)
            
            return SystemMetrics(
                total_positions=portfolio_summary.get("total_positions", 0),
                open_positions=portfolio_summary.get("total_open_positions", 0),
                closed_positions=portfolio_summary.get("total_positions", 0) - portfolio_summary.get("total_open_positions", 0),
                total_pnl=portfolio_summary.get("total_pnl", 0.0),
                win_rate=win_rate,
                risk_status=risk_summary.get("risk_status", {}).get("overall_status", "UNKNOWN"),
                ai_confidence=ai_confidence,
                system_health=system_health
            )
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return SystemMetrics(
                total_positions=0,
                open_positions=0,
                closed_positions=0,
                total_pnl=0.0,
                win_rate=0.0,
                risk_status="ERROR",
                ai_confidence=0.0,
                system_health="ERROR"
            )
    
    def run_manual_cycle(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """Run a manual trading cycle."""
        if mode is None:
            mode = get_current_mode()
        
        try:
            logger.info(f"Running manual trading cycle for {mode} mode")
            
            # Execute complete trading cycle
            cycle_results = execute_complete_cycle(mode)
            
            return {
                "success": True,
                "message": "Manual trading cycle completed",
                "data": {
                    "positions_opened": cycle_results.positions_opened,
                    "positions_closed": cycle_results.positions_closed,
                    "total_pnl": cycle_results.total_pnl,
                    "duration_seconds": cycle_results.duration_seconds
                }
            }
            
        except Exception as e:
            logger.error(f"Error running manual cycle: {e}")
            return {
                "success": False,
                "message": f"Manual trading cycle failed: {e}",
                "error": str(e)
            }
    
    def analyze_market(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """Analyze current market conditions."""
        if mode is None:
            mode = get_current_mode()
        
        try:
            # Get market features
            market_features = self._get_market_features()
            
            # Analyze market regime
            market_regime = self.enhanced_ensemble.analyze_market_regime(market_features)
            
            # Get portfolio summary
            portfolio_summary = get_portfolio_summary(mode)
            
            return {
                "success": True,
                "message": "Market analysis completed",
                "data": {
                    "market_regime": market_regime,
                    "market_features": market_features,
                    "portfolio_summary": portfolio_summary
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market: {e}")
            return {
                "success": False,
                "message": f"Market analysis failed: {e}",
                "error": str(e)
            }
    
    def get_ai_recommendations(self, symbol: str, mode: Optional[str] = None) -> Dict[str, Any]:
        """Get AI recommendations for a specific symbol."""
        if mode is None:
            mode = get_current_mode()
        
        try:
            # Get market features for symbol
            market_features = self._get_symbol_features(symbol)
            
            # Analyze for entry
            buy_signal = analyze_for_entry(symbol, market_features, mode)
            
            # Analyze for exit (if position exists)
            position = self.trading_cycle.position_manager.get_position_by_symbol(symbol, mode)
            sell_signal = None
            if position:
                sell_signal = analyze_for_exit(position, market_features, mode)
            
            return {
                "success": True,
                "message": "AI recommendations generated",
                "data": {
                    "symbol": symbol,
                    "buy_signal": buy_signal,
                    "sell_signal": sell_signal,
                    "market_features": market_features
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting AI recommendations for {symbol}: {e}")
            return {
                "success": False,
                "message": f"AI recommendations failed: {e}",
                "error": str(e)
            }
    
    def get_activity_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent activity logs."""
        try:
            # Get activity results from scheduler
            activity_results = self.activity_scheduler.get_activity_results(limit)
            
            # Convert to dictionary format
            logs = []
            for result in activity_results:
                logs.append({
                    "activity_type": result.activity_type.value,
                    "start_time": result.start_time.isoformat(),
                    "end_time": result.end_time.isoformat(),
                    "success": result.success,
                    "message": result.message,
                    "error": result.error
                })
            
            return logs
            
        except Exception as e:
            logger.error(f"Error getting activity logs: {e}")
            return []
    
    def _get_average_ai_confidence(self) -> float:
        """Get average AI confidence from recent analyses."""
        try:
            # This would integrate with actual AI confidence tracking
            # For now, return a simulated value
            import random
            return random.uniform(0.6, 0.9)
        except Exception as e:
            logger.error(f"Error getting AI confidence: {e}")
            return 0.0
    
    def _get_system_health(self) -> str:
        """Get overall system health status."""
        try:
            # Check all components
            status = self.get_phase2_status()
            
            if status.overall_status == "FULLY_OPERATIONAL":
                return "HEALTHY"
            elif status.overall_status == "PARTIALLY_OPERATIONAL":
                return "DEGRADED"
            else:
                return "UNHEALTHY"
                
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return "ERROR"
    
    def _calculate_win_rate(self, mode: str) -> float:
        """Calculate win rate from closed positions."""
        try:
            # Get closed positions from database
            # This would integrate with actual trade history
            # For now, return a simulated value
            import random
            return random.uniform(0.4, 0.7)
        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return 0.0
    
    def _get_market_features(self) -> Dict[str, Any]:
        """Get current market features."""
        # This would integrate with real market data in production
        # For now, return simulated data
        import random
        
        return {
            "market_regime": random.choice(["BULL", "BEAR", "SIDEWAYS", "NEUTRAL"]),
            "volatility": random.uniform(0.01, 0.05),
            "volume_trend": random.choice(["HIGH", "NORMAL", "LOW"]),
            "sector_performance": random.uniform(-0.1, 0.1),
            "news_impact": random.uniform(-0.5, 0.5)
        }
    
    def _get_symbol_features(self, symbol: str) -> Dict[str, Any]:
        """Get features for a specific symbol."""
        # This would integrate with real market data in production
        # For now, return simulated data
        import random
        
        return {
            "current_price": 150.0 + random.uniform(-10, 10),
            "rsi": random.uniform(20, 80),
            "macd": random.uniform(-2, 2),
            "sma_20": 150.0 + random.uniform(-5, 5),
            "sma_50": 150.0 + random.uniform(-8, 8),
            "bollinger_position": random.uniform(0, 1),
            "volume_ratio": random.uniform(0.5, 2.0),
            "atr": random.uniform(0.01, 0.05),
            "sentiment_score": random.uniform(-1, 1),
            "fundamental_score": random.uniform(0.3, 0.8)
        }
    
    def validate_integration(self) -> bool:
        """Validate the Phase 2 integration."""
        try:
            # Check all components
            status = self.get_phase2_status()
            
            if not status.enhanced_ensemble:
                logger.error("Enhanced AI Ensemble validation failed")
                return False
            
            if not status.trading_cycle:
                logger.error("Trading Cycle validation failed")
                return False
            
            if not status.integration:
                logger.error("Integration validation failed")
                return False
            
            logger.info("Phase 2 integration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Phase 2 integration validation error: {e}")
            return False

# Global Phase 2 integration instance
_phase2_integration: Optional[Phase2Integration] = None

def get_phase2_integration() -> Phase2Integration:
    """Get the global Phase 2 integration instance."""
    global _phase2_integration
    if _phase2_integration is None:
        _phase2_integration = Phase2Integration()
    return _phase2_integration

def initialize_phase2() -> bool:
    """Initialize Phase 2 components."""
    return get_phase2_integration().initialize_phase2()

def start_phase2() -> bool:
    """Start Phase 2 operations."""
    return get_phase2_integration().start_phase2()

def stop_phase2() -> bool:
    """Stop Phase 2 operations."""
    return get_phase2_integration().stop_phase2()

def get_phase2_status() -> Phase2Status:
    """Get Phase 2 status."""
    return get_phase2_integration().get_phase2_status()

def get_system_metrics(mode: Optional[str] = None) -> SystemMetrics:
    """Get system metrics."""
    return get_phase2_integration().get_system_metrics(mode)

def run_manual_cycle(mode: Optional[str] = None) -> Dict[str, Any]:
    """Run manual trading cycle."""
    return get_phase2_integration().run_manual_cycle(mode)

def analyze_market(mode: Optional[str] = None) -> Dict[str, Any]:
    """Analyze market conditions."""
    return get_phase2_integration().analyze_market(mode)

def get_ai_recommendations(symbol: str, mode: Optional[str] = None) -> Dict[str, Any]:
    """Get AI recommendations for a symbol."""
    return get_phase2_integration().get_ai_recommendations(symbol, mode)

def get_activity_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Get activity logs."""
    return get_phase2_integration().get_activity_logs(limit)
