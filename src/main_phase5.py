"""
Phase 5 Main Entry Point

This module serves as the main entry point for Phase 5: Adaptive Configuration
and Self-Learning System.
"""

import logging
import asyncio
import argparse
from typing import Optional
from datetime import datetime

# Import Phase 5 components
from src.workflows.phase5_integration import (
    get_phase5_integration, initialize_phase5, start_adaptive_learning,
    stop_adaptive_learning, get_phase5_status, get_optimization_summary,
    manual_optimization, get_parameter_recommendations
)
from src.adaptive.self_learning_engine import OptimizationObjective
from src.config.mode_manager import set_mode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/phase5_main.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class Phase5Main:
    """Main class for Phase 5 operations."""
    
    def __init__(self, mode: str = "DEMO"):
        self.mode = mode
        self.phase5_integration = None
        self.is_running = False
        
        logger.info(f"Phase 5 Main initialized for {mode} mode")
    
    async def initialize(self) -> bool:
        """Initialize Phase 5 system."""
        try:
            logger.info("Initializing Phase 5 system...")
            
            # Set mode
            set_mode(self.mode)
            
            # Initialize Phase 5 integration
            self.phase5_integration = get_phase5_integration(self.mode)
            
            if not await self.phase5_integration.initialize():
                logger.error("Failed to initialize Phase 5 integration")
                return False
            
            logger.info("Phase 5 system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Phase 5 system: {e}")
            return False
    
    async def start(self) -> bool:
        """Start Phase 5 system."""
        try:
            logger.info("Starting Phase 5 system...")
            
            if not self.phase5_integration:
                logger.error("Phase 5 integration not initialized")
                return False
            
            # Start adaptive learning
            if not await self.phase5_integration.start_adaptive_learning():
                logger.error("Failed to start adaptive learning")
                return False
            
            self.is_running = True
            logger.info("Phase 5 system started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting Phase 5 system: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop Phase 5 system."""
        try:
            logger.info("Stopping Phase 5 system...")
            
            if self.phase5_integration:
                await self.phase5_integration.stop_adaptive_learning()
            
            self.is_running = False
            logger.info("Phase 5 system stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping Phase 5 system: {e}")
            return False
    
    async def run(self) -> bool:
        """Run Phase 5 system."""
        try:
            logger.info("Running Phase 5 system...")
            
            if not self.is_running:
                logger.error("Phase 5 system not started")
                return False
            
            # Run the system (this will run the background loops)
            # The system runs continuously in the background
            logger.info("Phase 5 system is running in background mode")
            
            # Keep the system running
            while self.is_running:
                await asyncio.sleep(60)  # Check every minute
            
            return True
            
        except Exception as e:
            logger.error(f"Error running Phase 5 system: {e}")
            return False
    
    async def get_status(self) -> dict:
        """Get system status."""
        try:
            if not self.phase5_integration:
                return {"error": "Phase 5 integration not initialized"}
            
            return await self.phase5_integration.get_system_status()
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {"error": str(e)}
    
    async def get_optimization_summary(self) -> dict:
        """Get optimization summary."""
        try:
            if not self.phase5_integration:
                return {"error": "Phase 5 integration not initialized"}
            
            return await self.phase5_integration.get_optimization_summary()
            
        except Exception as e:
            logger.error(f"Error getting optimization summary: {e}")
            return {"error": str(e)}
    
    async def trigger_optimization(self, objective: str = None) -> dict:
        """Trigger manual optimization."""
        try:
            if not self.phase5_integration:
                return {"error": "Phase 5 integration not initialized"}
            
            # Parse objective
            opt_objective = None
            if objective:
                try:
                    opt_objective = OptimizationObjective(objective.upper())
                except ValueError:
                    return {"error": f"Invalid objective: {objective}"}
            
            return await self.phase5_integration.manual_optimization(opt_objective)
            
        except Exception as e:
            logger.error(f"Error triggering optimization: {e}")
            return {"error": str(e)}
    
    async def get_recommendations(self) -> list:
        """Get parameter recommendations."""
        try:
            if not self.phase5_integration:
                return []
            
            return await self.phase5_integration.get_parameter_recommendations()
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Phase 5: Adaptive Configuration and Self-Learning System")
    parser.add_argument("--mode", choices=["DEMO", "LIVE"], default="DEMO", help="Trading mode")
    parser.add_argument("--action", choices=["start", "stop", "status", "optimize", "recommendations"], 
                       default="start", help="Action to perform")
    parser.add_argument("--objective", choices=[
        "MAXIMIZE_SHARPE_RATIO", "MAXIMIZE_PROFIT_FACTOR", "MINIMIZE_DRAWDOWN",
        "MAXIMIZE_WIN_RATE", "MAXIMIZE_TOTAL_RETURN", "BALANCED_PERFORMANCE"
    ], help="Optimization objective")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    
    args = parser.parse_args()
    
    # Create Phase 5 main instance
    phase5_main = Phase5Main(mode=args.mode)
    
    try:
        if args.action == "start":
            # Initialize and start
            if not await phase5_main.initialize():
                logger.critical("Failed to initialize Phase 5 system")
                return
            
            if not await phase5_main.start():
                logger.critical("Failed to start Phase 5 system")
                return
            
            if args.daemon:
                # Run as daemon
                logger.info("Running Phase 5 system as daemon...")
                await phase5_main.run()
            else:
                # Run for a short time and show status
                logger.info("Phase 5 system started. Press Ctrl+C to stop.")
                await asyncio.sleep(5)  # Run for 5 seconds
                status = await phase5_main.get_status()
                logger.info(f"System Status: {status}")
        
        elif args.action == "stop":
            # Stop the system
            await phase5_main.stop()
            logger.info("Phase 5 system stopped")
        
        elif args.action == "status":
            # Get status
            if not await phase5_main.initialize():
                logger.error("Failed to initialize for status check")
                return
            
            status = await phase5_main.get_status()
            logger.info("Phase 5 System Status:")
            for key, value in status.items():
                if isinstance(value, dict):
                    logger.info(f"  {key.capitalize()}:")
                    for sub_key, sub_value in value.items():
                        logger.info(f"    {sub_key.capitalize()}: {sub_value}")
                else:
                    logger.info(f"  {key.capitalize()}: {value}")
        
        elif args.action == "optimize":
            # Trigger optimization
            if not await phase5_main.initialize():
                logger.error("Failed to initialize for optimization")
                return
            
            result = await phase5_main.trigger_optimization(args.objective)
            logger.info(f"Optimization Result: {result}")
        
        elif args.action == "recommendations":
            # Get recommendations
            if not await phase5_main.initialize():
                logger.error("Failed to initialize for recommendations")
                return
            
            recommendations = await phase5_main.get_recommendations()
            logger.info(f"Parameter Recommendations ({len(recommendations)}):")
            for i, rec in enumerate(recommendations):
                logger.info(f"  {i+1}. {rec['parameter_name']}: {rec['current_value']:.4f} → {rec['recommended_value']:.4f}")
                logger.info(f"     Confidence: {rec['confidence']:.2f}, Reasoning: {rec['reasoning']}")
    
    except KeyboardInterrupt:
        logger.info("Phase 5 system interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await phase5_main.stop()
        logger.info("Phase 5 system shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
