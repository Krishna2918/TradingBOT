#!/usr/bin/env python3
"""
Comprehensive test to verify all AI models and components are working properly
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

async def test_all_models():
    """Test all AI models and components"""
    try:
        logger.info("🧪 Testing All AI Models and Components...")
        
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
            logger.info("✅ All AI components imported successfully")
        except Exception as e:
            logger.error(f"❌ Failed to import AI components: {e}")
            return False
        
        # Test 2: Initialize MasterOrchestrator
        logger.info("Test 2: Initializing MasterOrchestrator...")
        try:
            orchestrator = MasterOrchestrator()
            logger.info("✅ MasterOrchestrator initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MasterOrchestrator: {e}")
            return False
        
        # Test 3: Test market data fetching
        logger.info("Test 3: Testing market data fetching...")
        try:
            import yfinance as yf
            import pandas as pd
            
            # Test with a simple stock
            ticker = yf.Ticker("TD.TO")
            hist = ticker.history(period='5d', interval='1d')
            
            if hist.empty:
                logger.warning("⚠️ No market data returned from Yahoo Finance")
            else:
                logger.info(f"✅ Market data fetched: {len(hist)} days of data")
                logger.info(f"Latest close price: ${hist['Close'].iloc[-1]:.2f}")
        except Exception as e:
            logger.error(f"❌ Failed to fetch market data: {e}")
            return False
        
        # Test 4: Test AI decision pipeline
        logger.info("Test 4: Testing AI decision pipeline...")
        try:
            # Create sample market data
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta
            
            # Generate sample data
            dates = pd.date_range(start=datetime.now() - timedelta(days=5), end=datetime.now(), freq='D')
            sample_data = pd.DataFrame({
                'Open': np.random.uniform(80, 85, len(dates)),
                'High': np.random.uniform(85, 90, len(dates)),
                'Low': np.random.uniform(75, 80, len(dates)),
                'Close': np.random.uniform(80, 85, len(dates)),
                'Volume': np.random.randint(1000000, 5000000, len(dates))
            }, index=dates)
            
            # Run AI decision pipeline
            decision = await orchestrator.run_decision_pipeline(sample_data)
            
            if decision:
                logger.info(f"✅ AI Decision: {decision.action} with confidence {decision.confidence:.2f}")
                logger.info(f"Reasoning: {decision.reasoning}")
            else:
                logger.warning("⚠️ No decision returned from AI pipeline")
        except Exception as e:
            logger.error(f"❌ Failed to run AI decision pipeline: {e}")
            return False
        
        # Test 5: Test clean state manager
        logger.info("Test 5: Testing clean state manager...")
        try:
            from src.dashboard.clean_state_manager import state_manager, Trade
            from datetime import datetime
            
            # Start new session
            session_id = state_manager.start_new_session(10000, "DEMO")
            logger.info(f"✅ New session started: {session_id}")
            
            # Add a test trade
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
            logger.info("✅ Test trade added successfully")
            
            # Check state
            state = state_manager.get_current_state()
            logger.info(f"Current capital: ${state['current_capital']:,.2f}")
            logger.info(f"Positions: {len(state['positions'])}")
            logger.info(f"Trades: {len(state['trades'])}")
            
        except Exception as e:
            logger.error(f"❌ Failed to test clean state manager: {e}")
            return False
        
        # Test 6: Test AI trading engine
        logger.info("Test 6: Testing AI trading engine...")
        try:
            from src.dashboard.ai_trading_engine import ai_engine
            
            # Initialize engine
            if ai_engine.initialize():
                logger.info("✅ AI Trading Engine initialized")
            else:
                logger.error("❌ AI Trading Engine initialization failed")
                return False
            
            # Test one trading cycle
            await ai_engine.execute_trading_cycle()
            logger.info("✅ AI trading cycle executed successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to test AI trading engine: {e}")
            return False
        
        # Test 7: Test dashboard services
        logger.info("Test 7: Testing dashboard services...")
        try:
            from src.dashboard.services import get_demo_price, get_random_tsx_stock, is_market_open
            
            # Test demo price
            symbol = get_random_tsx_stock()
            price = get_demo_price(symbol)
            market_open = is_market_open()
            
            logger.info(f"✅ Random stock: {symbol}")
            logger.info(f"✅ Demo price: ${price:.2f}")
            logger.info(f"✅ Market open: {market_open}")
            
        except Exception as e:
            logger.error(f"❌ Failed to test dashboard services: {e}")
            return False
        
        # Test 8: Test mode manager
        logger.info("Test 8: Testing mode manager...")
        try:
            from src.config.mode_manager import get_mode_manager
            
            mode_manager = get_mode_manager()
            current_mode = mode_manager.get_current_mode()
            
            logger.info(f"✅ Current mode: {current_mode}")
            
        except Exception as e:
            logger.error(f"❌ Failed to test mode manager: {e}")
            return False
        
        # Test 9: Test system monitor
        logger.info("Test 9: Testing system monitor...")
        try:
            from src.monitoring.system_monitor import SystemMonitor
            
            monitor = SystemMonitor()
            health = monitor.get_health_status()
            
            logger.info(f"✅ System health: {health['status']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to test system monitor: {e}")
            return False
        
        logger.info("🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_all_models())
    if success:
        print("\n✅ ALL MODELS AND COMPONENTS ARE WORKING CORRECTLY!")
    else:
        print("\n❌ SOME MODELS OR COMPONENTS HAVE ISSUES")
        sys.exit(1)
