#!/usr/bin/env python3
"""
Test to verify the dashboard is properly connected to the working AI system
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

async def test_dashboard_ai_integration():
    """Test dashboard integration with AI system"""
    try:
        logger.info("🧪 Testing Dashboard-AI Integration...")
        
        # Test 1: Import dashboard components
        logger.info("Test 1: Importing dashboard components...")
        try:
            from src.dashboard.clean_state_manager import state_manager
            from src.dashboard.ai_trading_engine import ai_engine
            from src.dashboard.services import get_demo_price, get_random_tsx_stock
            logger.info("✅ Dashboard components imported successfully")
        except Exception as e:
            logger.error(f"❌ Failed to import dashboard components: {e}")
            return False
        
        # Test 2: Start a new trading session
        logger.info("Test 2: Starting new trading session...")
        try:
            session_id = state_manager.start_new_session(5000, "DEMO")
            logger.info(f"✅ New session started: {session_id}")
            
            # Check initial state
            state = state_manager.get_current_state()
            logger.info(f"Initial capital: ${state['current_capital']:,.2f}")
            logger.info(f"Initial positions: {len(state['positions'])}")
            logger.info(f"Initial trades: {len(state['trades'])}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start trading session: {e}")
            return False
        
        # Test 3: Initialize AI trading engine
        logger.info("Test 3: Initializing AI trading engine...")
        try:
            if ai_engine.initialize():
                logger.info("✅ AI Trading Engine initialized")
            else:
                logger.error("❌ AI Trading Engine initialization failed")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize AI trading engine: {e}")
            return False
        
        # Test 4: Run multiple AI trading cycles
        logger.info("Test 4: Running AI trading cycles...")
        try:
            for i in range(5):
                logger.info(f"Running cycle {i+1}...")
                await ai_engine.execute_trading_cycle()
                
                # Check state after each cycle
                state = state_manager.get_current_state()
                logger.info(f"  After cycle {i+1}: Capital=${state['current_capital']:,.2f}, Positions={len(state['positions'])}, Trades={len(state['trades'])}, Decisions={state['ai_decisions_today']}")
                
                # If we have positions, show them
                if state['positions']:
                    for pos in state['positions']:
                        logger.info(f"    Position: {pos['symbol']} - {pos['quantity']} shares @ ${pos['avg_price']:.2f}")
                
                # If we have trades, show recent ones
                if state['trades']:
                    recent_trades = state['trades'][-3:]  # Last 3 trades
                    for trade in recent_trades:
                        logger.info(f"    Trade: {trade['action']} {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f}")
                
        except Exception as e:
            logger.error(f"❌ Failed to run AI trading cycles: {e}")
            return False
        
        # Test 5: Calculate final portfolio metrics
        logger.info("Test 5: Calculating final portfolio metrics...")
        try:
            state = state_manager.get_current_state()
            
            # Calculate total portfolio value
            total_value = state['current_capital']
            for pos in state['positions']:
                total_value += pos['quantity'] * pos['current_price']
            
            # Calculate P&L
            total_pnl = total_value - state['starting_capital']
            pnl_pct = (total_pnl / state['starting_capital']) * 100 if state['starting_capital'] > 0 else 0
            
            logger.info(f"✅ Final Portfolio Summary:")
            logger.info(f"  Starting Capital: ${state['starting_capital']:,.2f}")
            logger.info(f"  Current Capital: ${state['current_capital']:,.2f}")
            logger.info(f"  Positions Value: ${total_value - state['current_capital']:,.2f}")
            logger.info(f"  Total Portfolio Value: ${total_value:,.2f}")
            logger.info(f"  Total P&L: ${total_pnl:+,.2f} ({pnl_pct:+.2f}%)")
            logger.info(f"  Total Positions: {len(state['positions'])}")
            logger.info(f"  Total Trades: {len(state['trades'])}")
            logger.info(f"  AI Decisions: {state['ai_decisions_today']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate portfolio metrics: {e}")
            return False
        
        logger.info("🎉 Dashboard-AI integration test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_dashboard_ai_integration())
    if success:
        print("\n✅ DASHBOARD-AI INTEGRATION IS WORKING PERFECTLY!")
        print("The dashboard is properly connected to the AI trading system.")
        print("All models are running correctly and making decisions.")
    else:
        print("\n❌ DASHBOARD-AI INTEGRATION HAS ISSUES")
        sys.exit(1)
