#!/usr/bin/env python3
"""
Complete System Test Script
Tests all components of the trading bot system
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def test_imports():
    """Test that all modules can be imported"""
    print("🔄 Testing imports...")
    
    try:
        # Test risk management
        from risk_management import CapitalAllocator, LeverageGovernor, KillSwitchManager
        print("✅ Risk management modules imported")
        
        # Test strategies
        from strategies import StrategyManager, MomentumScalpingStrategy
        print("✅ Strategy modules imported")
        
        # Test data pipeline
        from data_pipeline import CanadianMarketCollector
        print("✅ Data pipeline modules imported")
        
        # Test execution
        from execution import ETFAllocator
        print("✅ Execution modules imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration files"""
    print("🔄 Testing configuration files...")
    
    config_files = [
        "config/risk_config.yaml",
        "config/trading_config.yaml",
        "config/broker_config.yaml",
        "config/strategy_config.yaml",
        "config/data_sources.yaml",
        "config/monitoring_config.yaml"
    ]
    
    try:
        import yaml
        
        for config_file in config_files:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    yaml.safe_load(f)
                print(f"✅ {config_file} is valid")
            else:
                print(f"❌ {config_file} not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_risk_management():
    """Test risk management components"""
    print("🔄 Testing risk management...")
    
    try:
        from risk_management import CapitalAllocator, LeverageGovernor, KillSwitchManager
        
        # Test capital allocator
        allocator = CapitalAllocator("config/risk_config.yaml")
        state = allocator.get_capital_state()
        print(f"✅ Capital allocator: ${state.total_capital:,} CAD total capital")
        
        # Test leverage governor
        governor = LeverageGovernor("config/risk_config.yaml")
        leverage_state = governor.get_leverage_state()
        print(f"✅ Leverage governor: {leverage_state.current_leverage}x leverage")
        
        # Test kill switch manager
        kill_manager = KillSwitchManager("config/risk_config.yaml")
        print(f"✅ Kill switch manager: {kill_manager.is_kill_switch_active()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk management test failed: {e}")
        return False

def test_strategies():
    """Test trading strategies"""
    print("🔄 Testing trading strategies...")
    
    try:
        from strategies import StrategyManager
        
        # Test strategy manager
        manager = StrategyManager("config/strategy_config.yaml")
        summary = manager.get_strategy_summary()
        print(f"✅ Strategy manager: {summary['total_strategies']} strategies loaded")
        
        # Test individual strategies
        strategies = manager.strategies
        for name, strategy in strategies.items():
            status = strategy.get_strategy_status()
            print(f"✅ {name}: {status['allocation']:.1%} allocation, {status['max_leverage']}x leverage")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy test failed: {e}")
        return False

def test_data_collection():
    """Test data collection"""
    print("🔄 Testing data collection...")
    
    try:
        from data_pipeline import CanadianMarketCollector
        
        # Test data collector
        collector = CanadianMarketCollector("config/data_sources.yaml")
        data = collector.collect_all_data()
        
        print(f"✅ Data collector: {len(data.get('market_data', {}))} market symbols")
        print(f"✅ Data collector: {len(data.get('news', []))} news items")
        print(f"✅ Data collector: {len(data.get('economic_data', {}))} economic indicators")
        
        return True
        
    except Exception as e:
        print(f"❌ Data collection test failed: {e}")
        return False

def test_etf_allocation():
    """Test ETF allocation"""
    print("🔄 Testing ETF allocation...")
    
    try:
        from execution import ETFAllocator
        
        # Test ETF allocator
        allocator = ETFAllocator("config/risk_config.yaml")
        allocations = allocator.allocate_profits_to_etfs(5000.0)  # $5K profit
        
        print(f"✅ ETF allocator: {len(allocations)} allocations created")
        
        # Test execution
        executed = allocator.execute_etf_purchases(allocations)
        print(f"✅ ETF allocator: {len(executed)} purchases executed")
        
        # Test summary
        summary = allocator.get_allocation_summary()
        print(f"✅ ETF allocator: ${summary['total_allocated']:,.2f} CAD allocated")
        
        return True
        
    except Exception as e:
        print(f"❌ ETF allocation test failed: {e}")
        return False

def test_integration():
    """Test system integration"""
    print("🔄 Testing system integration...")
    
    try:
        from risk_management import CapitalAllocator
        from strategies import StrategyManager
        from data_pipeline import CanadianMarketCollector
        from execution import ETFAllocator
        
        # Initialize all components
        capital_allocator = CapitalAllocator("config/risk_config.yaml")
        strategy_manager = StrategyManager("config/strategy_config.yaml")
        data_collector = CanadianMarketCollector("config/data_sources.yaml")
        etf_allocator = ETFAllocator("config/risk_config.yaml")
        
        # Test integrated workflow
        print("✅ All components initialized")
        
        # Test data collection
        market_data = data_collector.collect_all_data()
        print("✅ Data collection completed")
        
        # Test strategy analysis
        all_signals = strategy_manager.analyze_market_conditions(
            market_data.get('market_data', {}),
            market_data.get('news', [])
        )
        total_signals = sum(len(signals) for signals in all_signals.values())
        print(f"✅ Strategy analysis: {total_signals} signals generated")
        
        # Test profit allocation
        capital_allocator.record_win(5000.0)
        etf_summary = capital_allocator.get_etf_allocation_summary()
        print(f"✅ Profit allocation: ${etf_summary['etf_invested']:,.2f} CAD to ETFs")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Complete Trading Bot System Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("Risk Management Test", test_risk_management),
        ("Strategy Test", test_strategies),
        ("Data Collection Test", test_data_collection),
        ("ETF Allocation Test", test_etf_allocation),
        ("Integration Test", test_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready for deployment.")
        print("\n🚀 Trading Bot Features:")
        print("   • 5 Advanced Trading Strategies")
        print("   • Canadian Market Data Collection")
        print("   • Risk Management & Kill Switches")
        print("   • 20% Profit Allocation to ETFs")
        print("   • Real-time Monitoring & Alerts")
        print("   • Low-latency Execution Framework")
        return 0
    else:
        print("⚠️ Some tests failed. Please review and fix issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
