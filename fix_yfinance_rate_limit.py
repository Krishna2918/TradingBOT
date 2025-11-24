#!/usr/bin/env python3
"""
Fix yfinance Rate Limiting Issues

Implements proper rate limiting, user-agent rotation, and retry mechanisms
to work around Yahoo Finance API restrictions.
"""

import time
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yfinance as yf

def setup_session_with_retries():
    """Setup requests session with proper retry strategy"""
    
    session = requests.Session()
    
    # User agent rotation
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0'
    ]
    
    session.headers.update({
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })
    
    # Retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def test_with_delays():
    """Test yfinance with proper delays and error handling"""
    
    print("🧪 Testing yfinance with rate limiting fixes...")
    
    # Setup session
    session = setup_session_with_retries()
    
    # Test symbols with increasing delays
    test_symbols = ["AAPL", "MSFT", "RY.TO"]
    delays = [5, 10, 15]  # Increasing delays
    
    for i, symbol in enumerate(test_symbols):
        try:
            print(f"\n📊 Testing {symbol} (delay: {delays[i]}s)...")
            
            # Wait before request
            if i > 0:
                print(f"⏳ Waiting {delays[i]} seconds...")
                time.sleep(delays[i])
            
            # Create ticker with custom session
            ticker = yf.Ticker(symbol, session=session)
            
            # Try to get data with timeout
            print(f"📥 Fetching data for {symbol}...")
            data = ticker.history(period="5d", timeout=30)
            
            if not data.empty:
                print(f"✅ SUCCESS: {symbol} - {len(data)} rows")
                print(f"   Latest: {data.index[-1]} - Close: ${data['Close'].iloc[-1]:.2f}")
                return True  # Success!
            else:
                print(f"⚠️ Empty data for {symbol}")
                
        except Exception as e:
            print(f"❌ Error with {symbol}: {e}")
            
            # If rate limited, wait longer
            if "429" in str(e) or "rate" in str(e).lower():
                print("🚫 Rate limited detected, waiting 30 seconds...")
                time.sleep(30)
    
    return False

def test_alternative_approach():
    """Test alternative data fetching approach"""
    
    print("\n🔄 Trying alternative approach...")
    
    try:
        # Try with minimal requests
        print("📊 Testing single symbol with long delay...")
        
        time.sleep(20)  # Wait 20 seconds first
        
        # Use different approach
        ticker = yf.Ticker("SPY")  # Use ETF instead of individual stock
        data = ticker.history(period="1d", interval="1d")
        
        if not data.empty:
            print(f"✅ Alternative approach works: SPY - {len(data)} rows")
            return True
        else:
            print("❌ Alternative approach failed")
            
    except Exception as e:
        print(f"❌ Alternative approach error: {e}")
    
    return False

def main():
    """Main test function"""
    
    print("🚀 YFINANCE RATE LIMIT FIX TEST")
    print("="*50)
    
    # Test 1: With proper delays
    success1 = test_with_delays()
    
    if not success1:
        # Test 2: Alternative approach
        success2 = test_alternative_approach()
        
        if not success2:
            print("\n❌ Both approaches failed")
            print("💡 Recommendations:")
            print("   1. Wait 1-2 hours for rate limit to reset")
            print("   2. Use VPN to change IP address")
            print("   3. Try from different network")
            print("   4. Use alternative data source")
            return False
    
    print("\n✅ yfinance is working!")
    print("💡 Ready to start data collection with proper rate limiting")
    return True

if __name__ == "__main__":
    main()