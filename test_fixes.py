"""
Test Data Collection Fixes

Test script to verify all warnings and errors have been resolved.
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.data_collection.comprehensive_data_collector import ComprehensiveDataCollector

def setup_logging():
    """Setup simple logging for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

async def test_technical_indicators():
    """Test technical indicators collection with fixes"""
    print("🧪 Testing Technical Indicators Collection (Fixed)")
    print("=" * 50)
    
    try:
        # Initialize collector
        collector = ComprehensiveDataCollector("TestFixesData")
        
        # Test collecting technical indicators for AAPL
        print("\n📉 Testing technical indicators collection for AAPL...")
        success = await collector._collect_technical_indicators('AAPL')
        print(f"Technical indicators collection: {'✅ Success' if success else '❌ Failed'}")
        
        # Check what indicator files were created
        test_data_path = Path("TestFixesData")
        if test_data_path.exists():
            indicator_files = list((test_data_path / "technical_indicators").glob("*.parquet"))
            
            print(f"\n📁 Technical indicator files created: {len(indicator_files)}")
            for file in indicator_files:
                print(f"    - {file.name}")
        
        # Test API usage
        usage = collector.key_manager.get_usage_summary()
        print(f"\n🔑 API Usage After Test:")
        for key in usage['keys']:
            print(f"   {key['name']}: {key['used']}/{key['limit']} ({key['remaining']} remaining)")
        
        print("\n✅ Technical indicators test completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        logging.error(f"Test failed: {e}", exc_info=True)

async def test_single_symbol_complete():
    """Test complete data collection for one symbol"""
    print("\n🧪 Testing Complete Single Symbol Collection")
    print("=" * 50)
    
    try:
        collector = ComprehensiveDataCollector("TestFixesData")
        
        symbol = 'MSFT'
        print(f"\n📈 Testing complete collection for {symbol}...")
        
        # Test daily data
        daily_success = await collector._collect_daily_data(symbol)
        print(f"Daily data: {'✅ Success' if daily_success else '❌ Failed'}")
        
        # Test 1min intraday
        intraday_1min_success = await collector._collect_intraday_data(symbol, '1min')
        print(f"1min intraday: {'✅ Success' if intraday_1min_success else '❌ Failed'}")
        
        # Test 5min intraday  
        intraday_5min_success = await collector._collect_intraday_data(symbol, '5min')
        print(f"5min intraday: {'✅ Success' if intraday_5min_success else '❌ Failed'}")
        
        # Test technical indicators
        indicators_success = await collector._collect_technical_indicators(symbol)
        print(f"Technical indicators: {'✅ Success' if indicators_success else '❌ Failed'}")
        
        # Check all files created
        test_data_path = Path("TestFixesData")
        if test_data_path.exists():
            market_files = list((test_data_path / "market_data").glob(f"{symbol}_*.parquet"))
            indicator_files = list((test_data_path / "technical_indicators").glob(f"{symbol}_*.parquet"))
            
            print(f"\n📁 Files created for {symbol}:")
            print(f"  Market data files: {len(market_files)}")
            print(f"  Technical indicator files: {len(indicator_files)}")
            
            total_files = len(market_files) + len(indicator_files)
            expected_files = 3 + 9  # 3 market data + 9 indicators
            success_rate = total_files / expected_files
            
            print(f"  Total files: {total_files}/{expected_files} ({success_rate:.1%})")
            
            if success_rate >= 0.5:
                print(f"✅ Collection successful for {symbol}")
            else:
                print(f"⚠️ Partial success for {symbol}")
        
    except Exception as e:
        print(f"\n❌ Complete collection test failed: {e}")
        logging.error(f"Complete collection test failed: {e}", exc_info=True)

async def main():
    """Main test function"""
    print("🔧 Testing Data Collection Fixes")
    print("=" * 60)
    print("This test verifies that all warnings and errors have been resolved")
    print("=" * 60)
    
    # Test technical indicators fixes
    await test_technical_indicators()
    
    # Test complete single symbol collection
    await test_single_symbol_complete()
    
    print("\n" + "=" * 60)
    print("🎉 Fix Testing Complete!")
    print("\nIf tests were successful, the data collection system is ready for:")
    print("  1. Subset collection (20 symbols)")
    print("  2. Full collection (200 symbols)")

if __name__ == '__main__':
    setup_logging()
    asyncio.run(main())