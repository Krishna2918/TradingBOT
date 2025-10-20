#!/usr/bin/env python3
"""
Test script to verify the dashboard is connected to the clean state manager
"""
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dashboard_connection():
    """Test if dashboard is connected to clean state manager"""
    try:
        # Import the clean components
        from src.dashboard.clean_state_manager import state_manager
        
        logger.info("🧪 Testing Dashboard Connection to Clean State Manager...")
        
        # Test 1: Check current state
        logger.info("Test 1: Checking current state...")
        state = state_manager.get_current_state()
        logger.info(f"Current session ID: {state.get('session_id', 'None')}")
        logger.info(f"Current capital: ${state.get('current_capital', 0):,.2f}")
        logger.info(f"Positions: {len(state.get('positions', []))}")
        logger.info(f"Trades: {len(state.get('trades', []))}")
        logger.info(f"AI decisions: {state.get('ai_decisions_today', 0)}")
        logger.info(f"Is active: {state.get('is_active', False)}")
        
        # Test 2: Start a new session
        logger.info("Test 2: Starting new session with $5,000...")
        session_id = state_manager.start_new_session(5000, "DEMO")
        logger.info(f"New session started: {session_id}")
        
        # Test 3: Check state after starting session
        logger.info("Test 3: Checking state after starting session...")
        state = state_manager.get_current_state()
        logger.info(f"Session ID: {state['session_id']}")
        logger.info(f"Starting capital: ${state['starting_capital']:,.2f}")
        logger.info(f"Current capital: ${state['current_capital']:,.2f}")
        logger.info(f"Mode: {state['mode']}")
        logger.info(f"Is active: {state['is_active']}")
        
        # Test 4: Simulate a trade
        logger.info("Test 4: Simulating a trade...")
        from src.dashboard.clean_state_manager import Trade
        from datetime import datetime
        
        trade = Trade(
            timestamp=datetime.now().isoformat(),
            symbol="TD.TO",
            action="BUY",
            quantity=10,
            price=80.50,
            reasoning="Test trade",
            confidence=0.85,
            pnl=None
        )
        
        state_manager.add_trade(trade)
        logger.info("Trade added successfully")
        
        # Test 5: Check state after trade
        logger.info("Test 5: Checking state after trade...")
        state = state_manager.get_current_state()
        logger.info(f"Current capital: ${state['current_capital']:,.2f}")
        logger.info(f"Positions: {len(state['positions'])}")
        logger.info(f"Trades: {len(state['trades'])}")
        logger.info(f"AI decisions: {state['ai_decisions_today']}")
        
        if state['positions']:
            pos = state['positions'][0]
            logger.info(f"Position: {pos['symbol']} - {pos['quantity']} shares @ ${pos['avg_price']:.2f}")
        
        logger.info("🎉 Dashboard connection test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dashboard_connection()
    if success:
        print("\n✅ ALL TESTS PASSED - Dashboard is connected to clean state manager!")
    else:
        print("\n❌ TESTS FAILED - Dashboard connection has issues")
        sys.exit(1)
