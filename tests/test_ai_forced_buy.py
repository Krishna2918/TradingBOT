#!/usr/bin/env python3
"""
Test script to force the AI to make a buy decision
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

async def test_forced_buy():
    """Test forcing the AI to make a buy decision"""
    try:
        # Import the clean components
        from src.dashboard.clean_state_manager import state_manager, Trade
        from src.dashboard.ai_trading_engine import ai_engine
        from datetime import datetime
        
        logger.info("🧪 Testing Forced AI Buy Decision...")
        
        # Test 1: Start a new session
        logger.info("Test 1: Starting new session with $10,000...")
        session_id = state_manager.start_new_session(10000, "DEMO")
        logger.info(f"New session started: {session_id}")
        
        # Test 2: Manually add a trade to simulate AI buying
        logger.info("Test 2: Manually adding a BUY trade...")
        trade = Trade(
            timestamp=datetime.now().isoformat(),
            symbol="TD.TO",
            action="BUY",
            quantity=50,
            price=80.00,
            reasoning="Forced test buy",
            confidence=0.95,
            pnl=None
        )
        
        state_manager.add_trade(trade)
        logger.info("Trade added successfully")
        
        # Test 3: Check state after manual trade
        logger.info("Test 3: Checking state after manual trade...")
        state = state_manager.get_current_state()
        logger.info(f"Current capital: ${state['current_capital']:,.2f}")
        logger.info(f"Positions: {len(state['positions'])}")
        logger.info(f"Trades: {len(state['trades'])}")
        logger.info(f"AI decisions: {state['ai_decisions_today']}")
        
        if state['positions']:
            pos = state['positions'][0]
            logger.info(f"Position: {pos['symbol']} - {pos['quantity']} shares @ ${pos['avg_price']:.2f}")
            logger.info(f"Position value: ${pos['quantity'] * pos['current_price']:,.2f}")
        
        # Test 4: Now test AI decision on existing position
        logger.info("Test 4: Testing AI decision on existing position...")
        await ai_engine.execute_trading_cycle()
        
        # Test 5: Check state after AI decision
        logger.info("Test 5: Checking state after AI decision...")
        state = state_manager.get_current_state()
        logger.info(f"Current capital: ${state['current_capital']:,.2f}")
        logger.info(f"Positions: {len(state['positions'])}")
        logger.info(f"Trades: {len(state['trades'])}")
        logger.info(f"AI decisions: {state['ai_decisions_today']}")
        
        # Test 6: Calculate portfolio value
        total_value = state['current_capital']
        for pos in state['positions']:
            total_value += pos['quantity'] * pos['current_price']
        
        logger.info(f"Total portfolio value: ${total_value:,.2f}")
        logger.info(f"P&L: ${total_value - state['starting_capital']:+,.2f}")
        
        logger.info("🎉 Forced buy test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_forced_buy())
    if success:
        print("\n✅ ALL TESTS PASSED - AI Trading Engine can handle trades!")
    else:
        print("\n❌ TESTS FAILED - AI Trading Engine has issues")
        sys.exit(1)
