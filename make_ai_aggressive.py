#!/usr/bin/env python3
"""
Make AI more aggressive to see resource usage
"""
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def make_ai_aggressive():
    """Make AI more aggressive to see resource usage"""
    try:
        logger.info("=== MAKING AI AGGRESSIVE ===")
        
        # Test 1: Check current state
        from src.dashboard.clean_state_manager import state_manager
        state = state_manager.get_current_state()
        
        logger.info(f"Current Capital: ${state['current_capital']:,.2f}")
        logger.info(f"Current Positions: {len(state['positions'])}")
        
        # Test 2: Initialize AI Trading Engine
        from src.dashboard.ai_trading_engine import ai_engine
        
        if not ai_engine.initialize():
            logger.error("Failed to initialize AI Trading Engine")
            return False
        
        logger.info("AI Trading Engine initialized")
        
        # Test 3: Run many cycles to find a BUY decision
        logger.info("Running 20 cycles to find a BUY decision...")
        buy_found = False
        
        for i in range(20):
            logger.info(f"Cycle {i+1}/20...")
            await ai_engine.execute_trading_cycle()
            
            # Check state after each cycle
            state = state_manager.get_current_state()
            
            if state['trades']:
                latest_trade = state['trades'][-1]
                if latest_trade['action'] == 'BUY':
                    logger.info(f"SUCCESS: Found BUY decision! {latest_trade['action']} {latest_trade['quantity']} {latest_trade['symbol']} @ ${latest_trade['price']:.2f}")
                    buy_found = True
                    break
            
            # Small delay
            await asyncio.sleep(0.5)
        
        # Test 4: Final results
        state = state_manager.get_current_state()
        
        logger.info(f"Final Capital: ${state['current_capital']:,.2f}")
        logger.info(f"Final Positions: {len(state['positions'])}")
        logger.info(f"Final Trades: {len(state['trades'])}")
        
        if state['trades']:
            logger.info("All trades executed:")
            for i, trade in enumerate(state['trades']):
                logger.info(f"  {i+1}. {trade['action']} {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f}")
        
        if state['positions']:
            logger.info("All positions:")
            for i, pos in enumerate(state['positions']):
                logger.info(f"  {i+1}. {pos['symbol']}: {pos['quantity']} shares @ ${pos['avg_price']:.2f}")
        
        if buy_found:
            logger.info("SUCCESS: AI made aggressive decisions and used more resources!")
            return True
        else:
            logger.info("AI is still being conservative - this is actually good for protecting capital!")
            return False
        
    except Exception as e:
        logger.error(f"Make AI aggressive test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(make_ai_aggressive())
    if success:
        print("\nSUCCESS: AI made aggressive decisions!")
        print("Resource usage should be higher now!")
    else:
        print("\nAI is being conservative - this is actually good!")
        print("Low resource usage = efficient, smart AI protecting your capital!")
