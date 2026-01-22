"""
Test Data Collection

Simple test script to verify the data collection system works
by collecting data for a small number of symbols.
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

async def test_single_symbol_collection():
    """Test collecting data for a single symbol"""
    print("🧪 Testing Single Symbol Data Collection")
    print("=" * 50)
    
    try:
        # Initialize collector
        collector = ComprehensiveDataCollector("TestData")
        
        # Test collecting daily data for AAPL
        print("\n📈 Testing daily data collection for AAPL...")
        success = await collector._collect_daily_data('AAPL')
        print(f"Daily data collection: {'✅ Success' if success else '❌ Failed'}")
        
        # Test collecting 1min intraday data for AAPL
        print("\n📊 Testing 1min intraday data collection for AAPL...")
        success = await collector._collect_intraday_data('AAPL', '1min')
        print(f"1min intraday collection: {'✅ Success' if success else '❌ Failed'}")
        
        # Test collecting technical indicators for AAPL
        print("\n📉 Testing technical indicators collection for AAPL...")
        success = await collector._collect_technical_indicators('AAPL')
        print(f"Technical indicators collection: {'✅ Success' if success else '❌ Failed'}")
        
        # Check what files were created
        test_data_path = Path("TestData")
        if test_data_path.exists():
            market_data_files = list((test_data_path / "market_data").glob("*.parquet"))
            indicator_files = list((test_data_path / "technical_indicators").glob("*.parquet"))
            
            print(f"\n📁 Files created:")
            print(f"  Market data files: {len(market_data_files)}")
            print(f"  Technical indicator files: {len(indicator_files)}")
            
            for file in market_data_files:
                print(f"    - {file.name}")
            
            for file in indicator_files[:5]:  # Show first 5 indicator files
                print(f"    - {file.name}")
            
            if len(indicator_files) > 5:
                print(f"    ... and {len(indicator_files) - 5} more indicator files")
        
        print("\n✅ Single symbol test completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        logging.error(f"Test failed: {e}", exc_info=True)

async def test_key_manager():
    """Test the Alpha Vantage key manager"""
    print("\n🔑 Testing Alpha Vantage Key Manager")
    print("=" * 50)
    
    try:
        from src.data_collection.alpha_vantage_key_manager import get_alpha_vantage_key_manager
        
        key_manager = get_alpha_vantage_key_manager()
        
        # Get usage summary
        usage = key_manager.get_usage_summary()
        print(f"\nAPI Key Usage Summary:")
        print(f"Total used today: {usage['total_used']}/{usage['total_limit']}")
        print(f"Total remaining: {usage['total_remaining']}")
        
        for key in usage['keys']:
            print(f"  {key['name']} ({key['purpose']}): {key['used']}/{key['limit']} - {key['remaining']} remaining")
        
        # Test getting a key for market data
        key_info = key_manager.get_best_key('market_data')
        if key_info:
            key_name, api_key = key_info
            print(f"\nBest key for market data: {key_name}")
        else:
            print("\n❌ No available keys for market data")
        
        # Test making a simple request
        print(f"\n🌐 Testing API request...")
        data = key_manager.make_request('GLOBAL_QUOTE', {'symbol': 'AAPL'}, 'market_data')
        if data:
            print("✅ API request successful")
            if 'Global Quote' in data:
                quote = data['Global Quote']
                symbol = quote.get('01. symbol', 'N/A')
                price = quote.get('05. price', 'N/A')
                print(f"  Symbol: {symbol}, Price: ${price}")
        else:
            print("❌ API request failed")
        
    except Exception as e:
        print(f"\n❌ Key manager test failed: {e}")
        logging.error(f"Key manager test failed: {e}", exc_info=True)

async def main():
    """Main test function"""
    print("🚀 Starting Data Collection System Tests")
    print("=" * 60)
    
    # Test key manager first
    await test_key_manager()
    
    # Test single symbol collection
    await test_single_symbol_collection()
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")
    print("\nIf tests were successful, you can now run:")
    print("  python start_data_collection.py")
    print("to start the full 200-symbol collection process.")

if __name__ == '__main__':
    setup_logging()
    asyncio.run(main())