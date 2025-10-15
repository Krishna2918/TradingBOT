#!/usr/bin/env python3
"""
Comprehensive test to verify all models are active and working in the dashboard
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

async def test_all_models_active():
    """Test that all models are active and working"""
    try:
        logger.info("Testing All Models Are Active and Working...")
        
        # Test 1: Import all AI components
        logger.info("Test 1: Importing AI components...")
        try:
            from src.integration.master_orchestrator import MasterOrchestrator
            from src.ai.model_communication_hub import ModelCommunicationHub
            from src.ai.intelligent_model_selector import IntelligentModelSelector
            from src.ai.model_performance_learner import ModelPerformanceLearner
            from src.ai.performance_prediction_system import PerformancePredictionSystem
            from src.ai.market_condition_analyzer import MarketConditionAnalyzer
            from src.ai.cross_model_validator import CrossModelValidator
            logger.info("SUCCESS: All AI components imported successfully")
        except Exception as e:
            logger.error(f"FAILED: Failed to import AI components: {e}")
            return False
        
        # Test 2: Initialize MasterOrchestrator
        logger.info("Test 2: Initializing MasterOrchestrator...")
        try:
            orchestrator = MasterOrchestrator()
            logger.info("SUCCESS: MasterOrchestrator initialized successfully")
        except Exception as e:
            logger.error(f"FAILED: Failed to initialize MasterOrchestrator: {e}")
            return False
        
        # Test 3: Test clean state manager
        logger.info("Test 3: Testing clean state manager...")
        try:
            from src.dashboard.clean_state_manager import state_manager, Trade
            from datetime import datetime
            
            # Start new session
            session_id = state_manager.start_new_session(5000, "DEMO")
            logger.info(f"SUCCESS: New session started: {session_id}")
            
            # Check initial state
            state = state_manager.get_current_state()
            logger.info(f"Initial capital: ${state['current_capital']:,.2f}")
            logger.info(f"Positions: {len(state['positions'])}")
            logger.info(f"Trades: {len(state['trades'])}")
            logger.info(f"Is active: {state['is_active']}")
            
        except Exception as e:
            logger.error(f"FAILED: Clean state manager test failed: {e}")
            return False
        
        # Test 4: Test AI trading engine
        logger.info("Test 4: Testing AI trading engine...")
        try:
            from src.dashboard.ai_trading_engine import ai_engine
            
            # Initialize engine
            if ai_engine.initialize():
                logger.info("SUCCESS: AI Trading Engine initialized")
            else:
                logger.error("FAILED: AI Trading Engine initialization failed")
                return False
            
            # Test one trading cycle
            await ai_engine.execute_trading_cycle()
            logger.info("SUCCESS: AI trading cycle executed successfully")
            
        except Exception as e:
            logger.error(f"FAILED: AI trading engine test failed: {e}")
            return False
        
        # Test 5: Check state after AI cycle
        logger.info("Test 5: Checking state after AI cycle...")
        try:
            state = state_manager.get_current_state()
            logger.info(f"Capital after cycle: ${state['current_capital']:,.2f}")
            logger.info(f"Positions after cycle: {len(state['positions'])}")
            logger.info(f"Trades after cycle: {len(state['trades'])}")
            logger.info(f"AI decisions: {state['ai_decisions_today']}")
            
            # Show positions if any
            if state['positions']:
                for pos in state['positions']:
                    logger.info(f"Position: {pos['symbol']} - {pos['quantity']} shares @ ${pos['avg_price']:.2f}")
            
            # Show recent trades if any
            if state['trades']:
                for trade in state['trades'][-3:]:  # Last 3 trades
                    logger.info(f"Trade: {trade['action']} {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f}")
            
        except Exception as e:
            logger.error(f"FAILED: State check after AI cycle failed: {e}")
            return False
        
        # Test 6: Test dashboard services
        logger.info("Test 6: Testing dashboard services...")
        try:
            from src.dashboard.services import get_demo_price, get_random_tsx_stock, is_market_open
            
            # Test demo price
            symbol = get_random_tsx_stock()
            price = get_demo_price(symbol)
            market_open = is_market_open()
            
            logger.info(f"SUCCESS: Random stock: {symbol}")
            logger.info(f"SUCCESS: Demo price: ${price:.2f}")
            logger.info(f"SUCCESS: Market open: {market_open}")
            
        except Exception as e:
            logger.error(f"FAILED: Dashboard services test failed: {e}")
            return False
        
        # Test 7: Test mode manager
        logger.info("Test 7: Testing mode manager...")
        try:
            from src.config.mode_manager import get_mode_manager
            
            mode_manager = get_mode_manager()
            current_mode = mode_manager.get_current_mode()
            
            logger.info(f"SUCCESS: Current mode: {current_mode}")
            
        except Exception as e:
            logger.error(f"FAILED: Mode manager test failed: {e}")
            return False
        
        # Test 8: Test system monitor
        logger.info("Test 8: Testing system monitor...")
        try:
            from src.monitoring.system_monitor import SystemMonitor
            
            monitor = SystemMonitor()
            health = monitor.get_health_status()
            
            logger.info(f"SUCCESS: System health: {health['status']}")
            
        except Exception as e:
            logger.error(f"FAILED: System monitor test failed: {e}")
            return False
        
        logger.info("ALL TESTS COMPLETED SUCCESSFULLY!")
        logger.info("All models are active and working correctly!")
        return True
        
    except Exception as e:
        logger.error(f"Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_all_models_active())
    if success:
        print("\nSUCCESS: ALL MODELS ARE ACTIVE AND WORKING CORRECTLY!")
        print("The dashboard should be properly connected to the AI system.")
    else:
        print("\nFAILED: SOME MODELS OR COMPONENTS HAVE ISSUES")
        sys.exit(1)
