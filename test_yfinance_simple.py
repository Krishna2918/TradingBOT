#!/usr/bin/env python3
"""
Simple yfinance test to verify API connectivity
"""

import yfinance as yf
import pandas as pd

def test_yfinance():
    print("🧪 Testing yfinance connectivity...")
    
    # Test with US symbols first (more reliable)
    us_symbols = ["AAPL", "MSFT", "GOOGL"]
    
    for symbol in us_symbols:
        try:
            print(f"Testing {symbol}...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="5d")
            
            if not data.empty:
                print(f"✅ {symbol}: {len(data)} rows, latest: {data.index[-1]}")
            else:
                print(f"❌ {symbol}: No data")
                
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")
    
    # Test Canadian symbols
    print("\nTesting Canadian symbols...")
    ca_symbols = ["RY.TO", "SHOP.TO", "TD.TO"]
    
    for symbol in ca_symbols:
        try:
            print(f"Testing {symbol}...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="5d")
            
            if not data.empty:
                print(f"✅ {symbol}: {len(data)} rows, latest: {data.index[-1]}")
            else:
                print(f"❌ {symbol}: No data")
                
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")

if __name__ == "__main__":
    test_yfinance()