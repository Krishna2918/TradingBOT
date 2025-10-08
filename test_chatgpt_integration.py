#!/usr/bin/env python3
"""
Test script for ChatGPT Integration
Tests the ChatGPT API integration with the trading system
"""

import sys
import os
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ai.chatgpt_integration import ChatGPTIntegration
import json
import time

def test_chatgpt_integration():
    """Test the ChatGPT integration"""
    print("🤖 Testing ChatGPT Integration")
    print("=" * 50)
    
    # Configuration
    config = {
        'rate_limits': {
            'openai_api': {
                'requests_per_minute': 50,
                'requests_per_hour': 3000
            }
        }
    }
    
    # Initialize ChatGPT integration
    print("📡 Initializing ChatGPT Integration...")
    try:
        chatgpt = ChatGPTIntegration("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", config)
        print("✅ ChatGPT integration initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize ChatGPT: {e}")
        return
    
    # Test 1: API Status
    print("\n1️⃣ API Status Check:")
    status = chatgpt.get_api_status()
    print(json.dumps(status, indent=2))
    
    # Test 2: Market Analysis
    print("\n2️⃣ Market Analysis Test:")
    test_market_data = {
        'basic_market_data': {
            'TD.TO': None,  # Would contain actual market data
        },
        'news_sentiment': {
            'overall_sentiment': 0.2,
            'total_articles': 15,
            'symbols': {
                'TD.TO': {
                    'sentiment': 0.3,
                    'article_count': 5
                }
            }
        },
        'technical_indicators': {
            'TD.TO': {
                'sma_20': 100.5,
                'rsi_14': 65.2,
                'macd': 0.5
            }
        }
    }
    
    try:
        analysis = chatgpt.analyze_market_conditions(test_market_data)
        print("Market Analysis Result:")
        print(f"  Market Trend: {analysis.get('market_trend', 'N/A')}")
        print(f"  Overall Confidence: {analysis.get('overall_confidence', 0)}")
        print(f"  Risk Assessment: {analysis.get('risk_assessment', 'N/A')}")
        print(f"  Source: {analysis.get('source', 'N/A')}")
        print(f"  Model: {analysis.get('model', 'N/A')}")
        
        if analysis.get('recommendations'):
            print("  Recommendations:")
            for rec in analysis['recommendations']:
                print(f"    {rec.get('symbol', 'N/A')}: {rec.get('action', 'N/A')} (confidence: {rec.get('confidence', 0)})")
        
    except Exception as e:
        print(f"❌ Market analysis failed: {e}")
    
    # Test 3: Trading Decision
    print("\n3️⃣ Trading Decision Test:")
    try:
        decision = chatgpt.get_trading_decision('TD.TO', test_market_data)
        print("Trading Decision Result:")
        print(f"  Symbol: {decision.get('symbol', 'N/A')}")
        print(f"  Action: {decision.get('action', 'N/A')}")
        print(f"  Confidence: {decision.get('confidence', 0)}")
        print(f"  Reasoning: {decision.get('reasoning', 'N/A')}")
        print(f"  Entry Price: ${decision.get('entry_price', 0)}")
        print(f"  Stop Loss: ${decision.get('stop_loss', 0)}")
        print(f"  Take Profit: ${decision.get('take_profit', 0)}")
        print(f"  Position Size: {decision.get('position_size', 'N/A')}")
        print(f"  Source: {decision.get('source', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Trading decision failed: {e}")
    
    # Test 4: Rate Limiting
    print("\n4️⃣ Rate Limiting Test:")
    print(f"  Requests this minute: {status.get('requests_this_minute', 0)}")
    print(f"  Requests this hour: {status.get('requests_this_hour', 0)}")
    print(f"  Rate limit status: {status.get('rate_limit_status', False)}")
    
    print("\n✅ ChatGPT Integration Test Complete!")
    print("\n💡 ChatGPT is now integrated into the trading system:")
    print("  • Market analysis with GPT-4")
    print("  • Trading decision recommendations")
    print("  • Enhanced AI decision-making")
    print("  • Rate limiting and error handling")

if __name__ == "__main__":
    test_chatgpt_integration()
