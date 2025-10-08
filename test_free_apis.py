#!/usr/bin/env python3
"""
Test script for Free APIs Integration
Demonstrates News API, Alpha Vantage, and Reddit API integration
"""

import sys
import os
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_services import FreeAPIsIntegration, create_free_apis_config
import json
import time

def test_free_apis():
    """Test the free APIs integration"""
    print("🔑 Testing Free APIs Integration")
    print("=" * 50)
    
    # Create configuration
    config = create_free_apis_config()
    
    # Initialize integration
    print("📡 Initializing Free APIs Integration...")
    api_integration = FreeAPIsIntegration(config)
    
    # Test symbols
    symbols = ['TD.TO', 'RY.TO', 'SHOP.TO']
    
    print(f"\n🎯 Testing with symbols: {', '.join(symbols)}")
    print("-" * 50)
    
    # Test 1: API Status
    print("\n1️⃣ API Status Check:")
    status = api_integration.get_api_status()
    print(json.dumps(status, indent=2))
    
    # Test 2: News Sentiment
    print("\n2️⃣ News Sentiment Analysis:")
    news_data = api_integration.get_news_sentiment(symbols)
    print(f"Overall Sentiment: {news_data.get('overall_sentiment', 0):.3f}")
    print(f"Total Articles: {news_data.get('total_articles', 0)}")
    print(f"Demo Mode: {news_data.get('demo_mode', False)}")
    
    # Test 3: Technical Indicators
    print("\n3️⃣ Technical Indicators:")
    for symbol in symbols:
        indicators = api_integration.get_technical_indicators(symbol)
        print(f"{symbol}:")
        print(f"  SMA 20: {indicators.get('sma_20', 0):.2f}")
        print(f"  RSI 14: {indicators.get('rsi_14', 0):.2f}")
        print(f"  MACD: {indicators.get('macd', 0):.3f}")
        print(f"  Demo Mode: {indicators.get('demo_mode', False)}")
    
    # Test 4: Reddit Sentiment
    print("\n4️⃣ Reddit Sentiment:")
    reddit_data = api_integration.get_reddit_sentiment(symbols)
    print(f"Overall Sentiment: {reddit_data.get('overall_sentiment', 0):.3f}")
    print(f"Total Mentions: {reddit_data.get('total_mentions', 0)}")
    print(f"Demo Mode: {reddit_data.get('demo_mode', False)}")
    
    # Test 5: Comprehensive Data
    print("\n5️⃣ Comprehensive Data:")
    comprehensive_data = api_integration.get_comprehensive_data(symbols)
    print(f"Timestamp: {comprehensive_data.get('timestamp')}")
    print(f"Symbols: {comprehensive_data.get('symbols')}")
    
    print("\n✅ Free APIs Integration Test Complete!")
    print("\n💡 To get REAL data:")
    print("1. Get FREE API keys from:")
    print("   - News API: https://newsapi.org")
    print("   - Alpha Vantage: https://www.alphavantage.co/support/#api-key")
    print("   - Reddit API: https://www.reddit.com/prefs/apps")
    print("2. Update config/data_pipeline_config.yaml")
    print("3. Restart the trading bot")

if __name__ == "__main__":
    test_free_apis()
