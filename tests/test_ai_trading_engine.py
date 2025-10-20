#!/usr/bin/env python3
"""
Test script to verify the AI Trading Engine is working properly
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

async def test_ai_trading_engine():
    """Test the AI trading engine functionality"""
    try:
        # Import the clean components
        from src.dashboard.clean_state_manager import state_manager
        from src.dashboard.ai_trading_engine import ai_engine
        
        logger.info("🧪 Testing AI Trading Engine...")
        
        # Test 1: Initialize AI engine
        logger.info("Test 1: Initializing AI Trading Engine...")
        if ai_engine.initialize():
            logger.info("✅ AI Trading Engine initialized successfully")
        else:
            logger.error("❌ AI Trading Engine initialization failed")
            return False
        
        # Test 2: Start a new session
        logger.info("Test 2: Starting new trading session...")
        session_id = state_manager.start_new_session(10000, "DEMO")
        logger.info(f"✅ New session started: {session_id}")
        
        # Test 3: Check initial state
        logger.info("Test 3: Checking initial state...")
        state = state_manager.get_current_state()
        logger.info(f"Initial capital: ${state['current_capital']:,.2f}")
        logger.info(f"Positions: {len(state['positions'])}")
        logger.info(f"Trades: {len(state['trades'])}")
        
        # Test 4: Run one trading cycle
        logger.info("Test 4: Running one AI trading cycle...")
        await ai_engine.execute_trading_cycle()
        
        # Test 5: Check state after trading cycle
        logger.info("Test 5: Checking state after trading cycle...")
        state = state_manager.get_current_state()
        logger.info(f"Capital after cycle: ${state['current_capital']:,.2f}")
        logger.info(f"Positions after cycle: {len(state['positions'])}")
        logger.info(f"Trades after cycle: {len(state['trades'])}")
        logger.info(f"AI decisions today: {state['ai_decisions_today']}")
        
        # Test 6: Run multiple cycles
        logger.info("Test 6: Running 3 more trading cycles...")
        for i in range(3):
            logger.info(f"Running cycle {i+2}...")
            await ai_engine.execute_trading_cycle()
            
            state = state_manager.get_current_state()
            logger.info(f"  After cycle {i+2}: Capital=${state['current_capital']:,.2f}, Positions={len(state['positions'])}, Trades={len(state['trades'])}")
        
        # Final state
        logger.info("Final Results:")
        state = state_manager.get_current_state()
        logger.info(f"✅ Final capital: ${state['current_capital']:,.2f}")
        logger.info(f"✅ Total positions: {len(state['positions'])}")
        logger.info(f"✅ Total trades: {len(state['trades'])}")
        logger.info(f"✅ AI decisions: {state['ai_decisions_today']}")
        
        # Show positions if any
        if state['positions']:
            logger.info("Current positions:")
            for pos in state['positions']:
                logger.info(f"  {pos['symbol']}: {pos['quantity']} shares @ ${pos['avg_price']:.2f}")
        
        # Show recent trades if any
        if state['trades']:
            logger.info("Recent trades:")
            for trade in state['trades'][-5:]:  # Last 5 trades
                logger.info(f"  {trade['action']} {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f}")
        
        logger.info("🎉 AI Trading Engine test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_ai_trading_engine())
    if success:
        print("\n✅ ALL TESTS PASSED - AI Trading Engine is working correctly!")
    else:
        print("\n❌ TESTS FAILED - AI Trading Engine has issues")
        sys.exit(1)
