"""
Test Technical Indicators API Calls

Test the correct Alpha Vantage technical indicators API format
"""

import sys
import asyncio
import logging

# Add src to path
sys.path.append('src')

from src.data_collection.alpha_vantage_key_manager import get_alpha_vantage_key_manager

async def test_technical_indicators():
    """Test technical indicators API calls"""
    
    logging.basicConfig(level=logging.INFO)
    
    key_manager = get_alpha_vantage_key_manager()
    
    # Test RSI with correct parameters
    print("🧪 Testing RSI API call...")
    
    params = {
        'symbol': 'AAPL',
        'interval': '5min',
        'time_period': '14',
        'series_type': 'close'
    }
    
    print(f"Parameters: {params}")
    
    data = key_manager.make_request('RSI', params, 'premium_realtime')
    
    if data:
        print(f"✅ RSI API call successful!")
        print(f"Response keys: {list(data.keys())}")
        
        if 'Technical Analysis: RSI' in data:
            rsi_data = data['Technical Analysis: RSI']
            print(f"RSI data points: {len(rsi_data)}")
            
            # Show first few data points
            for i, (date, values) in enumerate(rsi_data.items()):
                if i < 3:
                    print(f"  {date}: {values}")
        else:
            print(f"⚠️ 'Technical Analysis: RSI' not found in response")
    else:
        print(f"❌ RSI API call failed")
    
    # Test SMA
    print(f"\n🧪 Testing SMA API call...")
    
    params = {
        'symbol': 'AAPL',
        'interval': '5min',
        'time_period': '20',
        'series_type': 'close'
    }
    
    data = key_manager.make_request('SMA', params, 'premium_realtime')
    
    if data:
        print(f"✅ SMA API call successful!")
        print(f"Response keys: {list(data.keys())}")
    else:
        print(f"❌ SMA API call failed")

if __name__ == '__main__':
    asyncio.run(test_technical_indicators())