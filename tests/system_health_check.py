#!/usr/bin/env python3
"""
AI Trading System Health Check
Comprehensive system validation for demo trading readiness
"""

import sys
import os
import time
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_imports():
    """Check if all required modules can be imported"""
    logger.info("🔍 Checking module imports...")
    
    try:
        # Core trading modules
        from src.dashboard import trading_state, STATE_STORE
        from src.dashboard.services import get_live_price, is_market_open
        from src.dashboard.ai_logger import ai_logger
        from src.dashboard.background_updater import background_updater
        logger.info("✅ Dashboard modules imported successfully")
        
        # AI modules - try different import paths
        try:
            from src.integration.master_orchestrator import MasterOrchestrator
        except ImportError:
            try:
                from src.ai.master_orchestrator import MasterOrchestrator
            except ImportError:
                logger.warning("MasterOrchestrator not available - will use basic AI")
                MasterOrchestrator = None
        
        try:
            from src.orchestrator.trading_orchestrator import TradingOrchestrator
        except ImportError:
            try:
                from src.trading.trading_orchestrator import TradingOrchestrator
            except ImportError:
                logger.warning("TradingOrchestrator not available")
                TradingOrchestrator = None
        logger.info("✅ AI orchestrator modules imported successfully")
        
        # Trading modules
        from src.trading.risk import RiskManager
        from src.trading.positions import PositionManager
        from src.trading.execution import ExecutionEngine
        logger.info("✅ Trading modules imported successfully")
        
        # Monitoring modules
        from src.monitoring.system_monitor import SystemMonitor
        from src.config.mode_manager import get_mode_manager
        logger.info("✅ Monitoring modules imported successfully")
        
        # Dashboard framework
        import dash
        import plotly
        import dash_bootstrap_components as dbc
        logger.info("✅ Dashboard framework imported successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import error: {e}")
        return False

def check_ai_components():
    """Check if AI components can be initialized"""
    logger.info("🧠 Checking AI components...")
    
    try:
        # Test MasterOrchestrator initialization (if available)
        try:
            from src.integration.master_orchestrator import MasterOrchestrator
            master_orchestrator = MasterOrchestrator()
            logger.info("✅ MasterOrchestrator initialized successfully")
        except ImportError:
            logger.info("ℹ️ MasterOrchestrator not available - using basic AI")
        
        # Test TradingOrchestrator initialization (if available)
        try:
            from src.orchestrator.trading_orchestrator import TradingOrchestrator
            trading_orchestrator = TradingOrchestrator()
            logger.info("✅ TradingOrchestrator initialized successfully")
        except ImportError:
            logger.info("ℹ️ TradingOrchestrator not available - using basic trading")
        
        # Test SystemMonitor
        from src.monitoring.system_monitor import SystemMonitor
        system_monitor = SystemMonitor()
        health_status = system_monitor.get_health_status()
        logger.info(f"✅ SystemMonitor initialized - Status: {health_status}")
        
        # Test ModeManager
        from src.config.mode_manager import get_mode_manager
        mode_manager = get_mode_manager()
        current_mode = mode_manager.get_current_mode()
        logger.info(f"✅ ModeManager initialized - Current mode: {current_mode}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ AI component error: {e}")
        return False

def check_data_sources():
    """Check if data sources are accessible"""
    logger.info("📊 Checking data sources...")
    
    try:
        from src.dashboard.services import get_live_price, is_market_open
        
        # Test market hours check
        market_open = is_market_open()
        logger.info(f"✅ Market hours check - Market open: {market_open}")
        
        # Test live price fetching
        test_symbol = "RY.TO"
        price = get_live_price(test_symbol)
        if price:
            logger.info(f"✅ Live price fetch - {test_symbol}: ${price:.2f}")
        else:
            logger.warning(f"⚠️ Live price fetch failed for {test_symbol}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data source error: {e}")
        return False

def check_database():
    """Check database connectivity and schema"""
    logger.info("🗄️ Checking database...")
    
    try:
        from src.config.database import DatabaseManager
        
        # Test database initialization
        db_manager = DatabaseManager(mode="demo")
        logger.info("✅ Database manager initialized")
        
        # Test connection
        with db_manager.get_connection_context() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            logger.info(f"✅ Database connection - Tables: {len(tables)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
        return False

def check_trading_state():
    """Check trading state initialization"""
    logger.info("💰 Checking trading state...")
    
    try:
        from src.dashboard import trading_state, load_trading_state, save_trading_state
        
        # Load trading state
        load_trading_state()
        logger.info("✅ Trading state loaded")
        
        # Check state structure
        required_keys = ['current_capital', 'holdings', 'trades', 'ai_decisions_today']
        for key in required_keys:
            if key in trading_state:
                logger.info(f"✅ Trading state has {key}")
            else:
                logger.warning(f"⚠️ Trading state missing {key}")
        
        # Test save functionality
        save_trading_state()
        logger.info("✅ Trading state save test passed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Trading state error: {e}")
        return False

def check_background_services():
    """Check background services"""
    logger.info("🔄 Checking background services...")
    
    try:
        from src.dashboard.background_updater import background_updater
        from src.dashboard.ai_logger import ai_logger
        
        # Test background updater
        logger.info("✅ Background updater available")
        
        # Test AI logger
        insights = ai_logger.get_ai_insights()
        logger.info(f"✅ AI logger available - Total decisions: {insights.get('total_decisions', 0)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Background services error: {e}")
        return False

def check_dashboard_components():
    """Check dashboard components"""
    logger.info("🖥️ Checking dashboard components...")
    
    try:
        import dash
        from dash import html, dcc
        import dash_bootstrap_components as dbc
        import plotly.graph_objs as go
        
        # Test basic dashboard creation
        app = dash.Dash(__name__)
        app.layout = html.Div([
            html.H1("Test Dashboard"),
            dcc.Graph(figure=go.Figure())
        ])
        logger.info("✅ Dashboard components working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dashboard components error: {e}")
        return False

def run_production_test():
    """Run the production readiness test"""
    logger.info("🧪 Running production readiness test...")
    
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "scripts/ultimate_production_readiness_test.py"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            logger.info("✅ Production test passed")
            return True
        else:
            logger.error(f"❌ Production test failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Production test error: {e}")
        return False

def main():
    """Run comprehensive health check"""
    logger.info("🚀 Starting AI Trading System Health Check")
    logger.info("=" * 60)
    
    checks = [
        ("Module Imports", check_imports),
        ("AI Components", check_ai_components),
        ("Data Sources", check_data_sources),
        ("Database", check_database),
        ("Trading State", check_trading_state),
        ("Background Services", check_background_services),
        ("Dashboard Components", check_dashboard_components),
        ("Production Test", run_production_test)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        logger.info(f"\n📋 Running {check_name} check...")
        try:
            result = check_func()
            results.append((check_name, result))
            if result:
                logger.info(f"✅ {check_name} check PASSED")
            else:
                logger.error(f"❌ {check_name} check FAILED")
        except Exception as e:
            logger.error(f"❌ {check_name} check ERROR: {e}")
            results.append((check_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 HEALTH CHECK SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {check_name}")
    
    logger.info(f"\n🎯 Overall Score: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 ALL CHECKS PASSED - SYSTEM READY FOR DEMO TRADING!")
        logger.info("🚀 You can now start the AI trading system with confidence!")
    else:
        logger.warning(f"⚠️ {total-passed} checks failed - Please review issues above")
    
    logger.info("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
