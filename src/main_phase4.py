"""
Main Entry Point for Phase 4: Collaborative AI Ensemble

This module provides the main entry point for Phase 4, integrating all
collaborative AI ensemble components with the existing system.
"""

import asyncio
import logging
import signal
import sys
from typing import Dict, Any, Optional
from datetime import datetime

from src.workflows.phase4_integration import (
    get_phase4_integration, initialize_phase4, start_phase4, stop_phase4,
    get_phase4_status, make_collaborative_decision, conduct_offhours_analysis_session,
    get_ensemble_metrics, run_ensemble_health_check, get_collaborative_decisions_history,
    validate_phase4_integration, CollaborativeDecision, AIEnsembleMetrics
)
from src.ai.offhours_discussion import DiscussionTopic
from src.config.mode_manager import get_current_mode, set_mode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/phase4.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class Phase4Main:
    """Main class for Phase 4 operations."""
    
    def __init__(self, mode: str = "DEMO"):
        self.mode = mode
        self.phase4_integration = get_phase4_integration(mode)
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Phase 4 Main initialized for {mode} mode")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_event.set()
    
    async def initialize(self) -> bool:
        """Initialize Phase 4 system."""
        try:
            logger.info("Initializing Phase 4 system...")
            
            # Initialize Phase 4 components
            if not await initialize_phase4(self.mode):
                logger.error("Failed to initialize Phase 4 components")
                return False
            
            # Validate integration
            if not await validate_phase4_integration(self.mode):
                logger.error("Phase 4 integration validation failed")
                return False
            
            logger.info("Phase 4 system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Phase 4 system: {e}")
            return False
    
    async def start(self) -> bool:
        """Start Phase 4 operations."""
        try:
            logger.info("Starting Phase 4 operations...")
            
            # Start Phase 4
            if not await start_phase4(self.mode):
                logger.error("Failed to start Phase 4 operations")
                return False
            
            self.is_running = True
            logger.info("Phase 4 operations started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting Phase 4 operations: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop Phase 4 operations."""
        try:
            logger.info("Stopping Phase 4 operations...")
            
            # Stop Phase 4
            if not await stop_phase4(self.mode):
                logger.error("Failed to stop Phase 4 operations")
                return False
            
            self.is_running = False
            logger.info("Phase 4 operations stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping Phase 4 operations: {e}")
            return False
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check."""
        try:
            logger.info("Running Phase 4 health check...")
            
            # Run ensemble health check
            health_check = await run_ensemble_health_check(self.mode)
            
            # Get status
            status = await get_phase4_status(self.mode)
            
            # Get metrics
            metrics = await get_ensemble_metrics(self.mode)
            
            health_summary = {
                "timestamp": datetime.now().isoformat(),
                "mode": self.mode,
                "status": status,
                "health_check": health_check,
                "metrics": metrics,
                "is_running": self.is_running
            }
            
            logger.info(f"Health check completed: {health_check.get('overall_health', 'UNKNOWN')}")
            return health_summary
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    async def demonstrate_collaborative_decision(self) -> Optional[CollaborativeDecision]:
        """Demonstrate collaborative decision making."""
        try:
            logger.info("Demonstrating collaborative decision making...")
            
            # Sample market data
            market_data = {
                "symbol": "AAPL",
                "price": 150.0,
                "volume": 1000000,
                "market_cap": 2500000000000,
                "sector": "Technology",
                "sentiment": 0.7,
                "technical_indicators": {
                    "rsi": 65,
                    "macd": 0.5,
                    "bollinger_position": 0.6
                },
                "news_sentiment": 0.8,
                "market_regime": "BULLISH"
            }
            
            # Make collaborative decision
            decision = await make_collaborative_decision("BUY_OR_SELL", market_data)
            
            if decision:
                logger.info(f"Collaborative decision: {decision.recommendation} (confidence: {decision.confidence:.2f})")
                logger.info(f"Reasoning: {decision.reasoning}")
                logger.info(f"Model contributions: {decision.model_contributions}")
            else:
                logger.error("Failed to make collaborative decision")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in collaborative decision demonstration: {e}")
            return None
    
    async def demonstrate_offhours_analysis(self) -> Optional[Dict[str, Any]]:
        """Demonstrate off-hours analysis."""
        try:
            logger.info("Demonstrating off-hours analysis...")
            
            # Conduct off-hours analysis
            session_summary = await conduct_offhours_analysis_session(DiscussionTopic.TRADE_ANALYSIS, self.mode)
            
            if session_summary:
                logger.info(f"Off-hours analysis completed: {session_summary['insights_count']} insights, {session_summary['improvements_count']} improvements")
                logger.info(f"Discussion summary: {session_summary['discussion_summary']}")
            else:
                logger.error("Failed to conduct off-hours analysis")
            
            return session_summary
            
        except Exception as e:
            logger.error(f"Error in off-hours analysis demonstration: {e}")
            return None
    
    async def run_demonstration(self) -> Dict[str, Any]:
        """Run comprehensive demonstration of Phase 4 capabilities."""
        try:
            logger.info("Running Phase 4 demonstration...")
            
            demonstration_results = {
                "timestamp": datetime.now().isoformat(),
                "mode": self.mode,
                "collaborative_decision": None,
                "offhours_analysis": None,
                "health_check": None,
                "metrics": None
            }
            
            # Run health check
            demonstration_results["health_check"] = await self.run_health_check()
            
            # Get metrics
            demonstration_results["metrics"] = await get_ensemble_metrics(self.mode)
            
            # Demonstrate collaborative decision
            demonstration_results["collaborative_decision"] = await self.demonstrate_collaborative_decision()
            
            # Demonstrate off-hours analysis
            demonstration_results["offhours_analysis"] = await self.demonstrate_offhours_analysis()
            
            # Get recent decisions history
            recent_decisions = get_collaborative_decisions_history(5, self.mode)
            demonstration_results["recent_decisions"] = [
                {
                    "topic": decision.topic,
                    "recommendation": decision.recommendation,
                    "confidence": decision.confidence,
                    "timestamp": decision.timestamp.isoformat()
                }
                for decision in recent_decisions
            ]
            
            logger.info("Phase 4 demonstration completed successfully")
            return demonstration_results
            
        except Exception as e:
            logger.error(f"Error in Phase 4 demonstration: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    async def run_continuous_monitoring(self):
        """Run continuous monitoring of Phase 4 system."""
        try:
            logger.info("Starting continuous monitoring...")
            
            while not self.shutdown_event.is_set():
                try:
                    # Run health check every 5 minutes
                    health_check = await self.run_health_check()
                    
                    # Log health status
                    overall_health = health_check.get("health_check", {}).get("overall_health", "UNKNOWN")
                    logger.info(f"System health: {overall_health}")
                    
                    # Check if system is unhealthy
                    if overall_health == "UNHEALTHY":
                        logger.error("System health is unhealthy, attempting recovery...")
                        # Could implement recovery logic here
                    
                    # Wait for next check or shutdown signal
                    try:
                        await asyncio.wait_for(self.shutdown_event.wait(), timeout=300)  # 5 minutes
                        break  # Shutdown signal received
                    except asyncio.TimeoutError:
                        continue  # Timeout, continue monitoring
                    
                except Exception as e:
                    logger.error(f"Error in continuous monitoring: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute before retrying
            
            logger.info("Continuous monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error in continuous monitoring: {e}")
    
    async def run(self):
        """Main run loop."""
        try:
            logger.info("Starting Phase 4 main loop...")
            
            # Initialize system
            if not await self.initialize():
                logger.error("Failed to initialize Phase 4 system")
                return False
            
            # Start operations
            if not await self.start():
                logger.error("Failed to start Phase 4 operations")
                return False
            
            # Run demonstration
            demonstration_results = await self.run_demonstration()
            logger.info("Demonstration results:")
            logger.info(f"  Health: {demonstration_results.get('health_check', {}).get('health_check', {}).get('overall_health', 'UNKNOWN')}")
            logger.info(f"  Collaborative Decision: {demonstration_results.get('collaborative_decision', {}).get('recommendation', 'NONE')}")
            logger.info(f"  Off-hours Analysis: {demonstration_results.get('offhours_analysis', {}).get('insights_count', 0)} insights")
            
            # Run continuous monitoring
            await self.run_continuous_monitoring()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in Phase 4 main loop: {e}")
            return False
        
        finally:
            # Cleanup
            await self.stop()
            logger.info("Phase 4 main loop completed")

async def main():
    """Main entry point."""
    try:
        # Get mode from command line or use default
        mode = "DEMO"
        if len(sys.argv) > 1:
            mode = sys.argv[1].upper()
            if mode not in ["DEMO", "LIVE"]:
                logger.error(f"Invalid mode: {mode}. Use DEMO or LIVE.")
                return 1
        
        # Set mode
        set_mode(mode)
        
        # Create and run Phase 4 main
        phase4_main = Phase4Main(mode)
        success = await phase4_main.run()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        return 0
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return 1

if __name__ == "__main__":
    # Run the main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
