"""
Main Entry Point for AI Trading System
=====================================

This is the main entry point for the AI Trading System that uses the
Master Orchestrator to coordinate all AI components for trading decisions.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import the master orchestrator
from src.integration.master_orchestrator import MasterOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class TradingSystem:
    """
    Main trading system that coordinates all AI components.
    """
    
    def __init__(self):
        """Initialize the trading system."""
        self.orchestrator = MasterOrchestrator()
        self.is_running = False
        
    async def start(self):
        """Start the trading system."""
        logger.info("Starting AI Trading System...")
        self.is_running = True
        
        try:
            # Initialize system
            await self._initialize_system()
            
            # Start main trading loop
            await self._trading_loop()
            
        except KeyboardInterrupt:
            logger.info("Received shutdown signal...")
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            await self._shutdown()
    
    async def _initialize_system(self):
        """Initialize the trading system components."""
        logger.info("Initializing system components...")
        
        # Check system status
        status = self.orchestrator.get_system_status()
        logger.info(f"System status: {status['system_health']}")
        
        logger.info("System initialization complete")
    
    async def _trading_loop(self):
        """Main trading loop."""
        logger.info("Starting main trading loop...")
        
        while self.is_running:
            try:
                # Generate sample market data for demonstration
                market_data = self._generate_sample_data()
                
                # Make trading decision
                decision = await self.orchestrator.make_trading_decision(market_data)
                
                # Log decision
                logger.info(f"Trading decision: {decision.action} "
                           f"(confidence: {decision.confidence:.3f}, "
                           f"position_size: {decision.position_size:.3f})")
                
                # Wait before next iteration
                await asyncio.sleep(60)  # 1 minute between decisions
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(10)  # Wait before retry
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample market data for demonstration."""
        # Generate 100 days of sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Generate realistic price data
        np.random.seed(42)  # For reproducible results
        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, 100)  # 0.1% daily return, 2% volatility
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate OHLCV data
        data = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000000, 5000000, 100)
        })
        
        return data
    
    async def _shutdown(self):
        """Shutdown the trading system."""
        logger.info("Shutting down trading system...")
        self.is_running = False
        self.orchestrator.shutdown()
        logger.info("Trading system shutdown complete")


async def main():
    """Main entry point."""
    try:
        # Create and start trading system
        trading_system = TradingSystem()
        await trading_system.start()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())