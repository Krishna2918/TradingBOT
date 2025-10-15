"""
Phase 2 Main Entry Point

This is the main entry point for Phase 2 of the AI Trading System.
It initializes and runs the enhanced AI ensemble, trading cycle, and activity scheduler.
"""

import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.workflows.phase2_integration import (
    get_phase2_integration, 
    initialize_phase2, 
    start_phase2, 
    stop_phase2,
    get_phase2_status,
    get_system_metrics,
    run_manual_cycle,
    analyze_market,
    get_ai_recommendations,
    get_activity_logs
)
from src.config.mode_manager import get_current_mode, set_mode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/phase2.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main Phase 2 entry point."""
    logger.info("=" * 60)
    logger.info("AI TRADING SYSTEM - PHASE 2 STARTUP")
    logger.info("=" * 60)
    
    try:
        # Set mode to DEMO for Phase 2
        set_mode('DEMO')
        current_mode = get_current_mode()
        logger.info(f"Operating in {current_mode} mode")
        
        # Initialize Phase 2
        logger.info("Initializing Phase 2 components...")
        if not initialize_phase2():
            logger.error("Failed to initialize Phase 2")
            return False
        
        # Get Phase 2 status
        status = get_phase2_status()
        logger.info(f"Phase 2 Status: {status.overall_status}")
        logger.info(f"  - Enhanced AI Ensemble: {'✓' if status.enhanced_ensemble else '✗'}")
        logger.info(f"  - Trading Cycle: {'✓' if status.trading_cycle else '✗'}")
        logger.info(f"  - Activity Scheduler: {'✓' if status.activity_scheduler else '✗'}")
        logger.info(f"  - Integration: {'✓' if status.integration else '✗'}")
        
        # Start Phase 2
        logger.info("Starting Phase 2 operations...")
        if not start_phase2():
            logger.error("Failed to start Phase 2")
            return False
        
        # Get system metrics
        metrics = get_system_metrics(current_mode)
        logger.info("System Metrics:")
        logger.info(f"  - Total Positions: {metrics.total_positions}")
        logger.info(f"  - Open Positions: {metrics.open_positions}")
        logger.info(f"  - Total P&L: ${metrics.total_pnl:.2f}")
        logger.info(f"  - Win Rate: {metrics.win_rate:.1%}")
        logger.info(f"  - Risk Status: {metrics.risk_status}")
        logger.info(f"  - AI Confidence: {metrics.ai_confidence:.2f}")
        logger.info(f"  - System Health: {metrics.system_health}")
        
        # Run initial market analysis
        logger.info("Running initial market analysis...")
        market_analysis = analyze_market(current_mode)
        if market_analysis['success']:
            logger.info(f"Market Analysis: {market_analysis['data']['market_regime']}")
        else:
            logger.warning(f"Market analysis failed: {market_analysis['message']}")
        
        # Run initial trading cycle
        logger.info("Running initial trading cycle...")
        cycle_result = run_manual_cycle(current_mode)
        if cycle_result['success']:
            logger.info(f"Trading Cycle: {cycle_result['data']['positions_opened']} opened, {cycle_result['data']['positions_closed']} closed")
        else:
            logger.warning(f"Trading cycle failed: {cycle_result['message']}")
        
        logger.info("=" * 60)
        logger.info("PHASE 2 STARTUP COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info("Phase 2 is now running with:")
        logger.info("  - Enhanced AI Ensemble for market analysis")
        logger.info("  - Complete Trading Cycle (BUY→HOLD→SELL)")
        logger.info("  - 24/7 Activity Scheduler")
        logger.info("  - Real-time monitoring and execution")
        logger.info("=" * 60)
        
        # Keep the system running
        logger.info("Phase 2 is now running. Press Ctrl+C to stop.")
        
        try:
            while True:
                time.sleep(60)  # Check every minute
                
                # Get updated status
                status = get_phase2_status()
                if status.overall_status != "FULLY_OPERATIONAL":
                    logger.warning(f"Phase 2 status degraded: {status.overall_status}")
                
                # Get updated metrics
                metrics = get_system_metrics(current_mode)
                logger.info(f"System Status: {metrics.system_health}, P&L: ${metrics.total_pnl:.2f}, Positions: {metrics.open_positions}")
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal...")
            
    except Exception as e:
        logger.error(f"Error in Phase 2 main: {e}")
        return False
    
    finally:
        # Stop Phase 2
        logger.info("Stopping Phase 2...")
        stop_phase2()
        logger.info("Phase 2 stopped successfully")
    
    return True

def test_phase2_components():
    """Test Phase 2 components individually."""
    logger.info("Testing Phase 2 components...")
    
    try:
        # Test initialization
        logger.info("Testing Phase 2 initialization...")
        if initialize_phase2():
            logger.info("✓ Phase 2 initialization successful")
        else:
            logger.error("✗ Phase 2 initialization failed")
            return False
        
        # Test status
        logger.info("Testing Phase 2 status...")
        status = get_phase2_status()
        logger.info(f"✓ Phase 2 status: {status.overall_status}")
        
        # Test system metrics
        logger.info("Testing system metrics...")
        metrics = get_system_metrics()
        logger.info(f"✓ System metrics: {metrics.system_health}")
        
        # Test market analysis
        logger.info("Testing market analysis...")
        market_analysis = analyze_market()
        if market_analysis['success']:
            logger.info("✓ Market analysis successful")
        else:
            logger.error("✗ Market analysis failed")
        
        # Test AI recommendations
        logger.info("Testing AI recommendations...")
        recommendations = get_ai_recommendations("AAPL")
        if recommendations['success']:
            logger.info("✓ AI recommendations successful")
        else:
            logger.error("✗ AI recommendations failed")
        
        # Test manual cycle
        logger.info("Testing manual trading cycle...")
        cycle_result = run_manual_cycle()
        if cycle_result['success']:
            logger.info("✓ Manual trading cycle successful")
        else:
            logger.error("✗ Manual trading cycle failed")
        
        logger.info("All Phase 2 component tests completed")
        return True
        
    except Exception as e:
        logger.error(f"Error testing Phase 2 components: {e}")
        return False

def get_phase2_info():
    """Get Phase 2 information and status."""
    logger.info("Phase 2 Information:")
    logger.info("=" * 40)
    
    try:
        # Get status
        status = get_phase2_status()
        logger.info(f"Overall Status: {status.overall_status}")
        logger.info(f"Enhanced AI Ensemble: {'Enabled' if status.enhanced_ensemble else 'Disabled'}")
        logger.info(f"Trading Cycle: {'Enabled' if status.trading_cycle else 'Disabled'}")
        logger.info(f"Activity Scheduler: {'Running' if status.activity_scheduler else 'Stopped'}")
        logger.info(f"Integration: {'Active' if status.integration else 'Inactive'}")
        
        # Get metrics
        metrics = get_system_metrics()
        logger.info(f"System Health: {metrics.system_health}")
        logger.info(f"Total Positions: {metrics.total_positions}")
        logger.info(f"Open Positions: {metrics.open_positions}")
        logger.info(f"Total P&L: ${metrics.total_pnl:.2f}")
        logger.info(f"Win Rate: {metrics.win_rate:.1%}")
        logger.info(f"AI Confidence: {metrics.ai_confidence:.2f}")
        
        # Get activity logs
        logs = get_activity_logs(5)
        logger.info(f"Recent Activities: {len(logs)}")
        for log in logs:
            logger.info(f"  - {log['activity_type']}: {log['message']}")
        
    except Exception as e:
        logger.error(f"Error getting Phase 2 info: {e}")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            # Run component tests
            success = test_phase2_components()
            sys.exit(0 if success else 1)
            
        elif command == "info":
            # Show Phase 2 information
            get_phase2_info()
            sys.exit(0)
            
        elif command == "start":
            # Start Phase 2
            success = main()
            sys.exit(0 if success else 1)
            
        else:
            print("Usage: python main_phase2.py [test|info|start]")
            print("  test  - Test Phase 2 components")
            print("  info  - Show Phase 2 information")
            print("  start - Start Phase 2 (default)")
            sys.exit(1)
    else:
        # Default: start Phase 2
        success = main()
        sys.exit(0 if success else 1)
