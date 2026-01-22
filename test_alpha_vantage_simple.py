#!/usr/bin/env python3
"""
Simple Alpha Vantage Test - Verify API is working
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_collection.alpha_vantage_collector import AlphaVantageCollector
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_alpha_vantage():
    """Test Alpha Vantage with known working symbols"""
    
    print("🔑 Testing Alpha Vantage API...")
    
    collector = AlphaVantageCollector()
    
    # Test with well-known symbols
    test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    for symbol in test_symbols:
        print(f"\\n📊 Testing {symbol}...")
        
        try:
            data, source = collector.fetch_daily_data(symbol)
            
            if data is not None and not data.empty:
                print(f"✅ {symbol}: {len(data)} records from {data.index.min()} to {data.index.max()}")
                print(f"   Latest: ${data['Close'].iloc[-1]:.2f}, Volume: {data['Volume'].iloc[-1]:,}")
            else:
                print(f"❌ {symbol}: No data returned")
                
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")
        
        # Small delay
        import time
        time.sleep(1)
    
    print("\\n🔑 Alpha Vantage test complete!")

if __name__ == "__main__":
    test_alpha_vantage()