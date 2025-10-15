"""
Phase 3 Main Entry Point

This is the main entry point for Phase 3 of the AI Trading System.
It initializes and runs the advanced AI models, autonomous trading system,
advanced risk management, and performance optimization.
"""

import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.workflows.phase3_integration import (
    get_phase3_integration,
    initialize_phase3,
    start_phase3,
    stop_phase3,
    get_phase3_status,
    get_system_health,
    run_comprehensive_optimization,
    perform_health_check,
    get_optimization_history,
    get_health_check_history,
    validate_phase3_integration
)
from src.config.mode_manager import get_current_mode, set_mode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/phase3.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main Phase 3 entry point."""
    logger.info("=" * 60)
    logger.info("AI TRADING SYSTEM - PHASE 3 STARTUP")
    logger.info("=" * 60)
    
    try:
        # Set mode to DEMO for Phase 3
        set_mode('DEMO')
        current_mode = get_current_mode()
        logger.info(f"Operating in {current_mode} mode")
        
        # Initialize Phase 3
        logger.info("Initializing Phase 3 components...")
        if not initialize_phase3():
            logger.error("Failed to initialize Phase 3")
            return False
        
        # Get Phase 3 status
        status = get_phase3_status()
        logger.info(f"Phase 3 Status: {status.overall_status}")
        logger.info(f"  - Advanced AI Models: {'✓' if status.advanced_ai_models else '✗'}")
        logger.info(f"  - Autonomous Trading: {'✓' if status.autonomous_trading else '✗'}")
        logger.info(f"  - Advanced Risk Management: {'✓' if status.advanced_risk_management else '✗'}")
        logger.info(f"  - Performance Optimization: {'✓' if status.performance_optimization else '✗'}")
        logger.info(f"  - Phase 2 Integration: {'✓' if status.phase2_integration else '✗'}")
        
        # Get system health
        system_health = get_system_health()
        logger.info(f"System Health: {system_health.overall_health}")
        
        # Start Phase 3
        logger.info("Starting Phase 3 operations...")
        if not start_phase3():
            logger.error("Failed to start Phase 3")
            return False
        
        # Run initial optimization
        logger.info("Running initial system optimization...")
        optimization_report = run_comprehensive_optimization()
        logger.info(f"Optimization completed: {optimization_report.overall_improvement:.2f}% improvement")
        
        # Perform initial health check
        logger.info("Performing initial health check...")
        health_check = perform_health_check()
        logger.info("Health check completed")
        
        logger.info("=" * 60)
        logger.info("PHASE 3 STARTUP COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info("Phase 3 is now running with:")
        logger.info("  - Advanced AI Models (Reinforcement Learning, Meta-Learning)")
        logger.info("  - Autonomous Trading System with Self-Learning")
        logger.info("  - Advanced Risk Management and Portfolio Optimization")
        logger.info("  - Performance Optimization and Scalability")
        logger.info("  - Comprehensive System Integration")
        logger.info("=" * 60)
        
        # Keep the system running
        logger.info("Phase 3 is now running. Press Ctrl+C to stop.")
        
        try:
            while True:
                time.sleep(300)  # Check every 5 minutes
                
                # Get updated status
                status = get_phase3_status()
                if status.overall_status != "FULLY_OPERATIONAL":
                    logger.warning(f"Phase 3 status degraded: {status.overall_status}")
                
                # Get updated health
                system_health = get_system_health()
                logger.info(f"System Status: {system_health.overall_health}")
                
                # Run optimization if needed
                if system_health.overall_health == "DEGRADED":
                    logger.info("System health degraded, running optimization...")
                    optimization_report = run_comprehensive_optimization()
                    logger.info(f"Optimization completed: {optimization_report.overall_improvement:.2f}% improvement")
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal...")
            
    except Exception as e:
        logger.error(f"Error in Phase 3 main: {e}")
        return False
    
    finally:
        # Stop Phase 3
        logger.info("Stopping Phase 3...")
        stop_phase3()
        logger.info("Phase 3 stopped successfully")
    
    return True

def test_phase3_components():
    """Test Phase 3 components individually."""
    logger.info("Testing Phase 3 components...")
    
    try:
        # Test initialization
        logger.info("Testing Phase 3 initialization...")
        if initialize_phase3():
            logger.info("✓ Phase 3 initialization successful")
        else:
            logger.error("✗ Phase 3 initialization failed")
            return False
        
        # Test status
        logger.info("Testing Phase 3 status...")
        status = get_phase3_status()
        logger.info(f"✓ Phase 3 status: {status.overall_status}")
        
        # Test system health
        logger.info("Testing system health...")
        system_health = get_system_health()
        logger.info(f"✓ System health: {system_health.overall_health}")
        
        # Test optimization
        logger.info("Testing system optimization...")
        optimization_report = run_comprehensive_optimization()
        logger.info(f"✓ System optimization: {optimization_report.overall_improvement:.2f}% improvement")
        
        # Test health check
        logger.info("Testing health check...")
        health_check = perform_health_check()
        logger.info("✓ Health check successful")
        
        # Test validation
        logger.info("Testing Phase 3 validation...")
        if validate_phase3_integration():
            logger.info("✓ Phase 3 validation successful")
        else:
            logger.error("✗ Phase 3 validation failed")
        
        logger.info("All Phase 3 component tests completed")
        return True
        
    except Exception as e:
        logger.error(f"Error testing Phase 3 components: {e}")
        return False

def get_phase3_info():
    """Get Phase 3 information and status."""
    logger.info("Phase 3 Information:")
    logger.info("=" * 40)
    
    try:
        # Get status
        status = get_phase3_status()
        logger.info(f"Overall Status: {status.overall_status}")
        logger.info(f"Advanced AI Models: {'Enabled' if status.advanced_ai_models else 'Disabled'}")
        logger.info(f"Autonomous Trading: {'Enabled' if status.autonomous_trading else 'Disabled'}")
        logger.info(f"Advanced Risk Management: {'Enabled' if status.advanced_risk_management else 'Disabled'}")
        logger.info(f"Performance Optimization: {'Enabled' if status.performance_optimization else 'Disabled'}")
        logger.info(f"Phase 2 Integration: {'Enabled' if status.phase2_integration else 'Disabled'}")
        
        # Get system health
        system_health = get_system_health()
        logger.info(f"System Health: {system_health.overall_health}")
        
        # Get optimization history
        optimization_history = get_optimization_history(5)
        logger.info(f"Recent Optimizations: {len(optimization_history)}")
        for opt in optimization_history:
            logger.info(f"  - {opt.timestamp}: {opt.overall_improvement:.2f}% improvement")
        
        # Get health check history
        health_check_history = get_health_check_history(5)
        logger.info(f"Recent Health Checks: {len(health_check_history)}")
        for health in health_check_history:
            logger.info(f"  - {health['timestamp']}: {health['overall_health']}")
        
    except Exception as e:
        logger.error(f"Error getting Phase 3 info: {e}")

def run_optimization_cycle():
    """Run a complete optimization cycle."""
    logger.info("Running optimization cycle...")
    
    try:
        # Perform health check
        logger.info("Performing health check...")
        health_check = perform_health_check()
        
        # Run optimization
        logger.info("Running system optimization...")
        optimization_report = run_comprehensive_optimization()
        
        # Log results
        logger.info(f"Optimization cycle completed:")
        logger.info(f"  - Overall improvement: {optimization_report.overall_improvement:.2f}%")
        logger.info(f"  - Performance optimizations: {len(optimization_report.performance_optimizations)}")
        logger.info(f"  - Risk optimizations: {len(optimization_report.risk_optimizations)}")
        logger.info(f"  - AI optimizations: {len(optimization_report.ai_optimizations)}")
        logger.info(f"  - Trading optimizations: {len(optimization_report.trading_optimizations)}")
        
        # Log recommendations
        logger.info("Recommendations:")
        for recommendation in optimization_report.recommendations:
            logger.info(f"  - {recommendation}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in optimization cycle: {e}")
        return False

def monitor_system():
    """Monitor system continuously."""
    logger.info("Starting system monitoring...")
    
    try:
        while True:
            # Get system health
            system_health = get_system_health()
            logger.info(f"System Health: {system_health.overall_health}")
            
            # Check if optimization is needed
            if system_health.overall_health == "DEGRADED":
                logger.info("System health degraded, running optimization...")
                run_optimization_cycle()
            
            # Wait before next check
            time.sleep(300)  # 5 minutes
            
    except KeyboardInterrupt:
        logger.info("System monitoring stopped")
    except Exception as e:
        logger.error(f"Error in system monitoring: {e}")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            # Run component tests
            success = test_phase3_components()
            sys.exit(0 if success else 1)
            
        elif command == "info":
            # Show Phase 3 information
            get_phase3_info()
            sys.exit(0)
            
        elif command == "optimize":
            # Run optimization cycle
            success = run_optimization_cycle()
            sys.exit(0 if success else 1)
            
        elif command == "monitor":
            # Monitor system
            monitor_system()
            sys.exit(0)
            
        elif command == "start":
            # Start Phase 3
            success = main()
            sys.exit(0 if success else 1)
            
        else:
            print("Usage: python main_phase3.py [test|info|optimize|monitor|start]")
            print("  test     - Test Phase 3 components")
            print("  info     - Show Phase 3 information")
            print("  optimize - Run optimization cycle")
            print("  monitor  - Monitor system continuously")
            print("  start    - Start Phase 3 (default)")
            sys.exit(1)
    else:
        # Default: start Phase 3
        success = main()
        sys.exit(0 if success else 1)
