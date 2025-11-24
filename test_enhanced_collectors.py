#!/usr/bin/env python3
"""
Test Enhanced Data Collectors

Tests the new enhanced collectors with rate limiting and multi-source fallback.
"""

import sys
import os
import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_collection.enhanced_collectors import MultiSourceDataCollector

def test_enhanced_collectors():
    """Test the enhanced collectors system"""
    
    print("🧪 TESTING ENHANCED DATA COLLECTORS")
    print("=" * 50)
    
    # Initialize collector
    collector = MultiSourceDataCollector()
    
    # Show source status
    print("\n📊 Data Source Status:")
    status = collector.get_source_status()
    for source, info in status.items():
        available = "✅" if info['available'] else "❌"
        healthy = "✅" if info['healthy'] else "❌"
        print(f"   {source.title()}: Available {available} | Healthy {healthy} | Priority {info['priority']}")
    
    # Test symbols
    test_symbols = ["SPY", "RY.TO", "TD.TO", "SHOP.TO"]
    
    print(f"\n🧪 Testing {len(test_symbols)} symbols...")
    
    results = {}
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}...")
        
        data, source = collector.fetch_data(symbol, period="5d", interval="1d")
        
        if data is not None and not data.empty:
            print(f"   ✅ SUCCESS from {source}: {len(data)} rows")
            print(f"   📅 Date range: {data.index.min()} to {data.index.max()}")
            print(f"   💰 Latest close: ${data['Close'].iloc[-1]:.2f}")
            results[symbol] = True
        else:
            print(f"   ❌ FAILED: No data from any source")
            results[symbol] = False
        
        # Small delay between symbols
        time.sleep(2)
    
    # Summary
    success_count = sum(results.values())
    total_count = len(results)
    success_rate = success_count / total_count * 100
    
    print(f"\n🎯 TEST RESULTS:")
    print(f"   Success: {success_count}/{total_count} ({success_rate:.1f}%)")
    
    if success_rate >= 75:
        print("   ✅ SYSTEM WORKING - Ready for data collection")
        return True
    else:
        print("   ⚠️ SYSTEM ISSUES - Some sources may need attention")
        
        # Refresh source health
        collector.refresh_source_health()
        return False

if __name__ == "__main__":
    print("🚀 TESTING ENHANCED DATA COLLECTORS")
    print("=" * 60)
    
    success = test_enhanced_collectors()
    
    if success:
        print("\n🎉 Enhanced collectors are working!")
        print("💡 You can now use the multi-source data collection system")
    else:
        print("\n⚠️ Some issues detected, but system may still be partially functional")
        print("💡 Check the logs above for specific source status")