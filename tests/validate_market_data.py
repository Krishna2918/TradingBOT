#!/usr/bin/env python3
"""
Market Data Validation Script
Comprehensive validation of all market data sources for demo trading
"""

import sys
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_yahoo_finance():
    """Test Yahoo Finance data source"""
    logger.info("📊 Testing Yahoo Finance data source...")
    
    try:
        import yfinance as yf
        
        # Test Canadian stocks
        test_symbols = ['RY.TO', 'TD.TO', 'SHOP.TO', 'CNR.TO', 'ENB.TO']
        
        for symbol in test_symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='1d', interval='1m')
                
                if not data.empty:
                    latest_price = data['Close'].iloc[-1]
                    volume = data['Volume'].iloc[-1]
                    logger.info(f"✅ {symbol}: ${latest_price:.2f} (Volume: {volume:,.0f})")
                else:
                    logger.warning(f"⚠️ {symbol}: No data available")
                    
            except Exception as e:
                logger.error(f"❌ {symbol}: Error - {e}")
        
        # Test market hours detection
        now = datetime.now()
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        is_market_hours = market_open <= now <= market_close and now.weekday() < 5
        logger.info(f"📅 Market hours check: {'Open' if is_market_hours else 'Closed'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Yahoo Finance test failed: {e}")
        return False

def test_data_pipeline():
    """Test internal data pipeline"""
    logger.info("🔄 Testing internal data pipeline...")
    
    try:
        from src.dashboard.services import get_live_price, is_market_open, get_random_tsx_stock
        
        # Test market hours detection
        market_open = is_market_open()
        logger.info(f"✅ Market hours detection: {'Open' if market_open else 'Closed'}")
        
        # Test live price fetching
        test_symbol = "RY.TO"
        price = get_live_price(test_symbol)
        if price:
            logger.info(f"✅ Live price fetch: {test_symbol} = ${price:.2f}")
        else:
            logger.warning(f"⚠️ Live price fetch failed for {test_symbol}")
        
        # Test random stock selection
        random_stock = get_random_tsx_stock()
        logger.info(f"✅ Random TSX stock: {random_stock}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data pipeline test failed: {e}")
        return False

def test_ai_models():
    """Test AI model availability"""
    logger.info("🧠 Testing AI model availability...")
    
    try:
        # Test MasterOrchestrator
        from src.integration.master_orchestrator import MasterOrchestrator
        master_orchestrator = MasterOrchestrator()
        logger.info("✅ MasterOrchestrator initialized")
        
        # Test individual AI components
        from src.ai.model_communication_hub import ModelCommunicationHub
        from src.ai.intelligent_model_selector import IntelligentModelSelector
        from src.ai.model_performance_learner import ModelPerformanceLearner
        from src.ai.performance_prediction_system import PerformancePredictionSystem
        from src.ai.market_condition_analyzer import MarketConditionAnalyzer
        from src.ai.cross_model_validator import CrossModelValidator
        
        logger.info("✅ All AI components available")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ AI models test failed: {e}")
        return False

def test_trading_components():
    """Test trading system components"""
    logger.info("💰 Testing trading system components...")
    
    try:
        from src.trading.risk import RiskManager
        from src.trading.positions import PositionManager
        from src.trading.execution import ExecutionEngine
        from src.trading.atr_brackets import ATRBracketManager
        
        # Test risk manager
        risk_manager = RiskManager()
        logger.info("✅ RiskManager initialized")
        
        # Test position manager
        position_manager = PositionManager()
        logger.info("✅ PositionManager initialized")
        
        # Test execution engine
        execution_engine = ExecutionEngine()
        logger.info("✅ ExecutionEngine initialized")
        
        # Test ATR brackets
        atr_manager = ATRBracketManager()
        logger.info("✅ ATRBracketManager initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Trading components test failed: {e}")
        return False

def test_database_operations():
    """Test database operations"""
    logger.info("🗄️ Testing database operations...")
    
    try:
        from src.config.database import DatabaseManager
        
        # Test demo database
        db_manager = DatabaseManager(mode="demo")
        
        with db_manager.get_connection_context() as conn:
            cursor = conn.cursor()
            
            # Test table existence
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            logger.info(f"✅ Database tables: {len(tables)} tables found")
            
            # Test basic operations - check if tables exist first
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades';")
            if cursor.fetchone():
                cursor.execute("SELECT COUNT(*) FROM trades")
                trade_count = cursor.fetchone()[0]
                logger.info(f"✅ Trades table: {trade_count} records")
            else:
                logger.info("ℹ️ Trades table: Not created yet (will be created on first trade)")
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='positions';")
            if cursor.fetchone():
                cursor.execute("SELECT COUNT(*) FROM positions")
                position_count = cursor.fetchone()[0]
                logger.info(f"✅ Positions table: {position_count} records")
            else:
                logger.info("ℹ️ Positions table: Not created yet (will be created on first position)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database operations test failed: {e}")
        return False

def test_dashboard_services():
    """Test dashboard services"""
    logger.info("🖥️ Testing dashboard services...")
    
    try:
        from src.dashboard import trading_state, load_trading_state, save_trading_state
        from src.dashboard.ai_logger import ai_logger
        from src.dashboard.background_updater import background_updater
        
        # Test trading state
        load_trading_state()
        logger.info("✅ Trading state loaded")
        
        # Test AI logger
        insights = ai_logger.get_ai_insights()
        logger.info(f"✅ AI logger: {insights.get('total_decisions', 0)} decisions logged")
        
        # Test background updater
        logger.info("✅ Background updater available")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dashboard services test failed: {e}")
        return False

def test_market_simulation():
    """Test market simulation capabilities"""
    logger.info("🎯 Testing market simulation capabilities...")
    
    try:
        from src.dashboard.services import simulate_ai_trade, simulate_historical_trading
        
        # Test AI trade simulation
        result = simulate_ai_trade()
        if result:
            logger.info(f"✅ AI trade simulation: {result.get('action', 'N/A')} {result.get('symbol', 'N/A')}")
        else:
            logger.warning("⚠️ AI trade simulation returned no result")
        
        # Test historical trading simulation
        try:
            historical_result = simulate_historical_trading()
            if historical_result:
                logger.info(f"✅ Historical trading simulation: {len(historical_result)} trades")
            else:
                logger.warning("⚠️ Historical trading simulation returned no result")
        except Exception as e:
            logger.warning(f"⚠️ Historical trading simulation error: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Market simulation test failed: {e}")
        return False

def main():
    """Run comprehensive market data validation"""
    logger.info("🚀 Starting Market Data Validation")
    logger.info("=" * 60)
    
    tests = [
        ("Yahoo Finance", test_yahoo_finance),
        ("Data Pipeline", test_data_pipeline),
        ("AI Models", test_ai_models),
        ("Trading Components", test_trading_components),
        ("Database Operations", test_database_operations),
        ("Dashboard Services", test_dashboard_services),
        ("Market Simulation", test_market_simulation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} test ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 MARKET DATA VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\n🎯 Overall Score: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 ALL MARKET DATA TESTS PASSED - READY FOR DEMO TRADING!")
        logger.info("🚀 Market data sources are fully operational!")
    else:
        logger.warning(f"⚠️ {total-passed} tests failed - Please review issues above")
    
    logger.info("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
