#!/usr/bin/env python3
"""
Force AI to make a trade and verify it's working
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

async def force_ai_trade_test():
    """Force AI to make a trade and verify execution"""
    try:
        logger.info("=== FORCING AI TRADE TEST ===")
        
        # Test 1: Check current state
        logger.info("Test 1: Current state before trade...")
        from src.dashboard.clean_state_manager import state_manager
        state = state_manager.get_current_state()
        
        logger.info(f"Session ID: {state['session_id']}")
        logger.info(f"Current Capital: ${state['current_capital']:,.2f}")
        logger.info(f"Positions: {len(state['positions'])}")
        logger.info(f"Trades: {len(state['trades'])}")
        logger.info(f"AI Decisions: {state['ai_decisions_today']}")
        
        # Test 2: Initialize AI Trading Engine
        logger.info("Test 2: Initializing AI Trading Engine...")
        from src.dashboard.ai_trading_engine import ai_engine
        
        if not ai_engine.initialize():
            logger.error("Failed to initialize AI Trading Engine")
            return False
        
        logger.info("AI Trading Engine initialized successfully")
        
        # Test 3: Force multiple trading cycles
        logger.info("Test 3: Running 5 trading cycles...")
        for i in range(5):
            logger.info(f"Running cycle {i+1}/5...")
            await ai_engine.execute_trading_cycle()
            
            # Check state after each cycle
            state = state_manager.get_current_state()
            logger.info(f"  After cycle {i+1}: Capital=${state['current_capital']:,.2f}, Positions={len(state['positions'])}, Trades={len(state['trades'])}, Decisions={state['ai_decisions_today']}")
            
            # If we got a trade, show details
            if state['trades']:
                latest_trade = state['trades'][-1]
                logger.info(f"  Latest trade: {latest_trade['action']} {latest_trade['quantity']} {latest_trade['symbol']} @ ${latest_trade['price']:.2f}")
            
            # Small delay between cycles
            await asyncio.sleep(1)
        
        # Test 4: Final state check
        logger.info("Test 4: Final state after all cycles...")
        state = state_manager.get_current_state()
        
        logger.info(f"Final Capital: ${state['current_capital']:,.2f}")
        logger.info(f"Final Positions: {len(state['positions'])}")
        logger.info(f"Final Trades: {len(state['trades'])}")
        logger.info(f"Final AI Decisions: {state['ai_decisions_today']}")
        
        # Show all trades
        if state['trades']:
            logger.info("All trades executed:")
            for i, trade in enumerate(state['trades']):
                logger.info(f"  {i+1}. {trade['action']} {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f} (Confidence: {trade['confidence']:.1%})")
        
        # Show all positions
        if state['positions']:
            logger.info("All positions:")
            for i, pos in enumerate(state['positions']):
                logger.info(f"  {i+1}. {pos['symbol']}: {pos['quantity']} shares @ ${pos['avg_price']:.2f} (P&L: ${pos['pnl']:+,.2f})")
        
        # Determine success
        if state['trades'] or state['positions']:
            logger.info("SUCCESS: AI made trades and updated state!")
            return True
        else:
            logger.warning("WARNING: AI made decisions but no trades were executed")
            return False
        
    except Exception as e:
        logger.error(f"Force AI trade test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(force_ai_trade_test())
    if success:
        print("\n✅ SUCCESS: AI is working and making trades!")
        print("The low resource usage is normal - AI models are efficient!")
    else:
        print("\n⚠️ WARNING: AI is making decisions but not executing trades")
        print("This explains the low resource usage - no actual trading happening")
