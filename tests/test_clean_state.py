#!/usr/bin/env python3
"""
Test the clean state manager to verify it's working correctly
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.dashboard.clean_state_manager import state_manager, Trade
from datetime import datetime

def test_clean_state():
    print("Testing Clean State Manager...")
    
    # Test 1: Start new session
    print("Test 1: Starting new session with $10,000...")
    session_id = state_manager.start_new_session(10000, "DEMO")
    print(f"Session started: {session_id}")
    
    # Test 2: Check initial state
    print("Test 2: Checking initial state...")
    state = state_manager.get_current_state()
    print(f"Current capital: ${state['current_capital']:,.2f}")
    print(f"Starting capital: ${state['starting_capital']:,.2f}")
    print(f"Positions: {len(state['positions'])}")
    print(f"Trades: {len(state['trades'])}")
    print(f"Is active: {state['is_active']}")
    
    # Test 3: Add a trade
    print("Test 3: Adding a test trade...")
    trade = Trade(
        timestamp=datetime.now().isoformat(),
        symbol="TD.TO",
        action="BUY",
        quantity=10,
        price=80.00,
        reasoning="Test trade",
        confidence=0.85,
        pnl=None
    )
    state_manager.add_trade(trade)
    
    # Test 4: Check state after trade
    print("Test 4: Checking state after trade...")
    state = state_manager.get_current_state()
    print(f"Current capital: ${state['current_capital']:,.2f}")
    print(f"Positions: {len(state['positions'])}")
    print(f"Trades: {len(state['trades'])}")
    print(f"AI decisions: {state['ai_decisions_today']}")
    
    if state['positions']:
        pos = state['positions'][0]
        print(f"Position: {pos['symbol']} - {pos['quantity']} shares @ ${pos['avg_price']:.2f}")
    
    print("Clean state manager test completed!")

if __name__ == "__main__":
    test_clean_state()
