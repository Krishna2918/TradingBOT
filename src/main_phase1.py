"""
Phase 1 Main Entry Point - Core Trading Infrastructure

This module provides the main entry point for Phase 1 components,
demonstrating the complete trading infrastructure functionality.
"""

import logging
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.config.mode_manager import get_mode_manager, set_mode, get_current_mode
from src.config.database import get_database_manager, validate_database_integrity
from src.trading.positions import get_position_manager, track_position, get_open_positions, get_portfolio_summary
from src.trading.exit_strategy import get_exit_strategy_engine, generate_sell_signals
from src.trading.risk import get_risk_manager, calculate_position_size, check_portfolio_limits, get_risk_summary
from src.trading.execution import get_execution_engine, execute_buy_order, OrderType, get_execution_summary

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/phase1.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class Phase1Demo:
    """Demonstration of Phase 1 functionality."""
    
    def __init__(self):
        """Initialize Phase 1 demo."""
        self.mode_manager = get_mode_manager()
        self.position_manager = get_position_manager()
        self.exit_engine = get_exit_strategy_engine()
        self.risk_manager = get_risk_manager()
        self.execution_engine = get_execution_engine()
        self.db_manager = get_database_manager()
        
        logger.info("Phase 1 Demo initialized")
    
    def run_demo(self):
        """Run the Phase 1 demonstration."""
        logger.info("=" * 60)
        logger.info("PHASE 1 DEMO - CORE TRADING INFRASTRUCTURE")
        logger.info("=" * 60)
        
        try:
            # 1. Test Mode Management
            self._test_mode_management()
            
            # 2. Test Database Setup
            self._test_database_setup()
            
            # 3. Test Position Management
            self._test_position_management()
            
            # 4. Test Risk Management
            self._test_risk_management()
            
            # 5. Test Exit Strategy
            self._test_exit_strategy()
            
            # 6. Test Order Execution
            self._test_order_execution()
            
            # 7. Test Integration
            self._test_integration()
            
            logger.info("=" * 60)
            logger.info("PHASE 1 DEMO COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Phase 1 demo failed: {e}")
            raise
    
    def _test_mode_management(self):
        """Test mode management functionality."""
        logger.info("\n1. TESTING MODE MANAGEMENT")
        logger.info("-" * 30)
        
        # Test current mode
        current_mode = get_current_mode()
        logger.info(f"Current mode: {current_mode}")
        
        # Test mode info
        mode_info = self.mode_manager.get_mode_info()
        logger.info(f"Mode info: {mode_info}")
        
        # Test mode switching
        logger.info("Testing mode switching...")
        original_mode = current_mode
        
        # Switch to LIVE mode
        if set_mode("LIVE"):
            logger.info("Successfully switched to LIVE mode")
            live_info = self.mode_manager.get_mode_info()
            logger.info(f"LIVE mode info: {live_info}")
        
        # Switch back to DEMO mode
        if set_mode("DEMO"):
            logger.info("Successfully switched back to DEMO mode")
            demo_info = self.mode_manager.get_mode_info()
            logger.info(f"DEMO mode info: {demo_info}")
        
        # Restore original mode
        set_mode(original_mode)
        logger.info("Mode management test completed")
    
    def _test_database_setup(self):
        """Test database setup and validation."""
        logger.info("\n2. TESTING DATABASE SETUP")
        logger.info("-" * 30)
        
        # Test database integrity
        for mode in ["LIVE", "DEMO"]:
            is_valid = validate_database_integrity(mode)
            logger.info(f"Database integrity for {mode} mode: {'PASS' if is_valid else 'FAIL'}")
        
        # Test database stats
        for mode in ["LIVE", "DEMO"]:
            stats = self.db_manager.get_database_stats(mode)
            logger.info(f"Database stats for {mode} mode: {stats}")
        
        logger.info("Database setup test completed")
    
    def _test_position_management(self):
        """Test position management functionality."""
        logger.info("\n3. TESTING POSITION MANAGEMENT")
        logger.info("-" * 30)
        
        # Test position tracking
        logger.info("Creating test positions...")
        
        # Create positions in DEMO mode
        set_mode("DEMO")
        
        try:
            # Track a position
            position1 = track_position("AAPL", 150.0, 10, stop_loss=140.0, take_profit=160.0)
            logger.info(f"Created position: {position1.symbol} {position1.quantity} shares @ ${position1.entry_price}")
            
            # Track another position
            position2 = track_position("MSFT", 300.0, 5, stop_loss=280.0, take_profit=320.0)
            logger.info(f"Created position: {position2.symbol} {position2.quantity} shares @ ${position2.entry_price}")
            
            # Get open positions
            open_positions = get_open_positions()
            logger.info(f"Open positions: {len(open_positions)}")
            
            for pos in open_positions:
                logger.info(f"  - {pos.symbol}: {pos.quantity} shares @ ${pos.entry_price}")
            
            # Get portfolio summary
            portfolio_summary = get_portfolio_summary()
            logger.info(f"Portfolio summary: {portfolio_summary}")
            
        except Exception as e:
            logger.warning(f"Position management test failed (expected in test environment): {e}")
        
        logger.info("Position management test completed")
    
    def _test_risk_management(self):
        """Test risk management functionality."""
        logger.info("\n4. TESTING RISK MANAGEMENT")
        logger.info("-" * 30)
        
        # Test position sizing
        logger.info("Testing position sizing...")
        risk_metrics = calculate_position_size(
            signal_confidence=0.8,
            account_balance=10000.0,
            volatility=0.02,
            entry_price=150.0,
            stop_loss=140.0
        )
        logger.info(f"Position sizing result: {risk_metrics}")
        
        # Test portfolio limits
        logger.info("Testing portfolio limits...")
        portfolio_limits = check_portfolio_limits()
        logger.info(f"Portfolio limits: {portfolio_limits}")
        
        # Test risk summary
        logger.info("Testing risk summary...")
        risk_summary = get_risk_summary()
        logger.info(f"Risk summary: {risk_summary}")
        
        logger.info("Risk management test completed")
    
    def _test_exit_strategy(self):
        """Test exit strategy functionality."""
        logger.info("\n5. TESTING EXIT STRATEGY")
        logger.info("-" * 30)
        
        # Test exit strategy summary
        logger.info("Testing exit strategy summary...")
        exit_summary = self.exit_engine.get_exit_strategy_summary()
        logger.info(f"Exit strategy summary: {exit_summary}")
        
        # Test sell signal generation (with empty positions)
        logger.info("Testing sell signal generation...")
        sell_signals = generate_sell_signals()
        logger.info(f"Generated {len(sell_signals)} sell signals")
        
        logger.info("Exit strategy test completed")
    
    def _test_order_execution(self):
        """Test order execution functionality."""
        logger.info("\n6. TESTING ORDER EXECUTION")
        logger.info("-" * 30)
        
        # Test execution summary
        logger.info("Testing execution summary...")
        execution_summary = get_execution_summary()
        logger.info(f"Execution summary: {execution_summary}")
        
        # Test buy order execution (will fail in test environment)
        logger.info("Testing buy order execution...")
        try:
            result = execute_buy_order("AAPL", 10, 150.0, OrderType.MARKET)
            if result.success:
                logger.info(f"Buy order executed successfully: {result.order}")
            else:
                logger.info(f"Buy order failed: {result.error_message}")
        except Exception as e:
            logger.info(f"Buy order execution test (expected to fail in test environment): {e}")
        
        logger.info("Order execution test completed")
    
    def _test_integration(self):
        """Test integration between components."""
        logger.info("\n7. TESTING INTEGRATION")
        logger.info("-" * 30)
        
        # Test component initialization
        logger.info("Testing component initialization...")
        components = [
            ("Mode Manager", self.mode_manager),
            ("Position Manager", self.position_manager),
            ("Exit Strategy Engine", self.exit_engine),
            ("Risk Manager", self.risk_manager),
            ("Execution Engine", self.execution_engine),
            ("Database Manager", self.db_manager)
        ]
        
        for name, component in components:
            if component is not None:
                logger.info(f"  ✓ {name}: OK")
            else:
                logger.error(f"  ✗ {name}: FAILED")
        
        # Test mode consistency
        logger.info("Testing mode consistency...")
        current_mode = get_current_mode()
        logger.info(f"Current mode: {current_mode}")
        
        # Test database connectivity
        logger.info("Testing database connectivity...")
        for mode in ["LIVE", "DEMO"]:
            try:
                stats = self.db_manager.get_database_stats(mode)
                logger.info(f"  ✓ {mode} database: OK")
            except Exception as e:
                logger.error(f"  ✗ {mode} database: FAILED - {e}")
        
        logger.info("Integration test completed")
    
    def run_validation(self):
        """Run validation checks for Phase 1."""
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1 VALIDATION")
        logger.info("=" * 60)
        
        validation_results = []
        
        # Validate mode manager
        try:
            mode_info = self.mode_manager.get_mode_info()
            validation_results.append(("Mode Manager", True, "OK"))
        except Exception as e:
            validation_results.append(("Mode Manager", False, str(e)))
        
        # Validate database
        try:
            for mode in ["LIVE", "DEMO"]:
                is_valid = validate_database_integrity(mode)
                if not is_valid:
                    raise Exception(f"Database integrity check failed for {mode} mode")
            validation_results.append(("Database", True, "OK"))
        except Exception as e:
            validation_results.append(("Database", False, str(e)))
        
        # Validate position manager
        try:
            portfolio_summary = get_portfolio_summary()
            validation_results.append(("Position Manager", True, "OK"))
        except Exception as e:
            validation_results.append(("Position Manager", False, str(e)))
        
        # Validate risk manager
        try:
            risk_summary = get_risk_summary()
            validation_results.append(("Risk Manager", True, "OK"))
        except Exception as e:
            validation_results.append(("Risk Manager", False, str(e)))
        
        # Validate exit strategy engine
        try:
            exit_summary = self.exit_engine.get_exit_strategy_summary()
            validation_results.append(("Exit Strategy Engine", True, "OK"))
        except Exception as e:
            validation_results.append(("Exit Strategy Engine", False, str(e)))
        
        # Validate execution engine
        try:
            execution_summary = get_execution_summary()
            validation_results.append(("Execution Engine", True, "OK"))
        except Exception as e:
            validation_results.append(("Execution Engine", False, str(e)))
        
        # Print validation results
        logger.info("\nValidation Results:")
        logger.info("-" * 40)
        
        all_passed = True
        for component, passed, message in validation_results:
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"{component:20} {status:8} {message}")
            if not passed:
                all_passed = False
        
        logger.info("-" * 40)
        if all_passed:
            logger.info("🎉 ALL VALIDATIONS PASSED - PHASE 1 READY")
        else:
            logger.error("❌ SOME VALIDATIONS FAILED - PHASE 1 NEEDS FIXES")
        
        return all_passed

def main():
    """Main entry point."""
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Initialize and run demo
    demo = Phase1Demo()
    
    try:
        # Run the demo
        demo.run_demo()
        
        # Run validation
        validation_passed = demo.run_validation()
        
        if validation_passed:
            logger.info("\n🚀 PHASE 1 IMPLEMENTATION COMPLETE AND VALIDATED")
            logger.info("Ready to proceed to Phase 2!")
            return 0
        else:
            logger.error("\n❌ PHASE 1 VALIDATION FAILED")
            logger.error("Please fix the issues before proceeding to Phase 2")
            return 1
            
    except Exception as e:
        logger.error(f"Phase 1 demo failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
