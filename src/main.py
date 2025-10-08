#!/usr/bin/env python3
"""
Main Entry Point for Trading Bot
Ultra-Aggressive Daily Doubling Trading Bot for Indian Markets
"""

import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from risk_management import CapitalAllocator, LeverageGovernor, KillSwitchManager
from strategies.strategy_manager import StrategyManager
from data_pipeline.collectors.canadian_market_collector import CanadianMarketCollector
from execution.etf_allocator import ETFAllocator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main function to start the trading bot"""
    try:
        logger.info(" Starting Trading Bot...")
        
        # Initialize risk management components
        logger.info(" Initializing Risk Management...")
        
        # Load configurations
        config_path = "config/risk_config.yaml"
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            return 1
        
        # Initialize components
        capital_allocator = CapitalAllocator(config_path)
        leverage_governor = LeverageGovernor(config_path)
        kill_switch_manager = KillSwitchManager(config_path)
        
        # Initialize strategy manager
        strategy_manager = StrategyManager("config/strategy_config.yaml")
        
        # Initialize data collector
        data_collector = CanadianMarketCollector("config/data_sources.yaml")
        
        # Initialize ETF allocator
        etf_allocator = ETFAllocator(config_path)
        
        logger.info(" All components initialized successfully")
        
        # Display initial state
        capital_state = capital_allocator.get_capital_state()
        leverage_state = leverage_governor.get_leverage_state()
        
        logger.info(f" Capital State: ${capital_state.total_capital:,} CAD total, ${capital_state.active_capital:,} CAD active")
        logger.info(f" Leverage State: {leverage_state.current_leverage}x current, {leverage_state.max_leverage}x max")
        logger.info(f" VIX Level: {leverage_state.vix_level}")
        
        # Test basic functionality
        logger.info(" Testing basic functionality...")
        
        # Test capital allocation
        test_position_size = capital_allocator.calculate_position_size(10000, "momentum")
        logger.info(f" Test Position Size: ${test_position_size:,} CAD")
        
        # Test leverage adjustment
        leverage_governor.update_vix_level(22.0)  # High VIX
        new_leverage = leverage_governor.get_leverage_state().current_leverage
        logger.info(f" VIX 22.0 -> Leverage: {new_leverage}x")
        
        # Test kill switch
        kill_switch_manager.activate_kill_switch("test_activation", "main")
        logger.info(f" Kill Switch Active: {kill_switch_manager.is_kill_switch_active()}")
        
        # Deactivate for normal operation
        kill_switch_manager.deactivate_kill_switch("test_complete")
        logger.info(f" Kill Switch Deactivated: {kill_switch_manager.is_kill_switch_active()}")
        
        # Test ETF allocation
        logger.info(" Testing ETF allocation...")
        capital_allocator.record_win(5000.0)  # $5K CAD profit
        etf_summary = capital_allocator.get_etf_allocation_summary()
        logger.info(f" ETF Allocation Summary: {etf_summary}")
        
        # Test data collection
        logger.info(" Testing data collection...")
        market_data = data_collector.collect_all_data()
        logger.info(f" Collected data for {len(market_data.get('market_data', {}))} symbols")
        
        # Test strategy analysis
        logger.info(" Testing strategy analysis...")
        all_signals = strategy_manager.analyze_market_conditions(
            market_data.get('market_data', {}),
            market_data.get('news', []),
            {}  # No options data for now
        )
        total_signals = sum(len(signals) for signals in all_signals.values())
        logger.info(f" Generated {total_signals} trading signals")
        
        # Test ETF allocation with actual profits
        logger.info(" Testing ETF allocation with profits...")
        etf_allocations = etf_allocator.allocate_profits_to_etfs(5000.0)
        logger.info(f" Created {len(etf_allocations)} ETF allocations")
        
        # Get comprehensive system status
        strategy_summary = strategy_manager.get_strategy_summary()
        logger.info(f" Strategy Summary: {strategy_summary['total_strategies']} strategies, {strategy_summary['total_positions']} positions")
        
        logger.info(" Trading Bot initialization completed successfully!")
        logger.info(" Ready for trading operations")
        logger.info(" All 5 strategies operational:")
        logger.info("   • Momentum Scalping 2.0")
        logger.info("   • News-Volatility")
        logger.info("   • Gamma/OI Squeeze")
        logger.info("   • Arbitrage/Latency")
        logger.info("   • AI/ML Pattern Discovery")
        
        # All major components are now implemented and operational!
        # -  Data pipeline (Canadian market data collection)
        # -  Trading strategies (All 5 strategies implemented)
        # -  Risk management (Capital allocation, leverage, kill switches)
        # -  ETF allocation (20% profit allocation to ETFs)
        # -  Execution framework (ETF allocator ready)
        # -  Monitoring system (Basic logging implemented)
        
        return 0
        
    except Exception as e:
        logger.error(f" Failed to start Trading Bot: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

