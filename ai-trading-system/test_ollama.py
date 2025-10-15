#!/usr/bin/env python3
"""
Test Ollama Integration
"""

from dotenv import load_dotenv
load_dotenv(".env")

import sys
sys.path.append('..')
from src.ai.ollama_client import ollama_client, ollama_trading_ai

print("Testing Ollama connection...")

if ollama_client.is_available():
    models = ollama_client.list_models()
    print(f"✅ Ollama ready: {len(models)} models available")
    print(f"Models: {[m['name'] for m in models[:3]]}")
    
    # Test a simple analysis
    print("\nTesting AI analysis...")
    result = ollama_trading_ai.analyze_market_data("AAPL", {"current_price": 150.0, "volume": 1000000})
    print(f"Analysis result: {result}")
else:
    print("❌ Ollama not available")
