#!/usr/bin/env python3
"""
Simple test script for AI trading system
"""

import sys
import os
sys.path.append('.')

from src.agents.ai_selector import AISelectorAgent, _build_pseudo_features
from src.ai.scoring import calculate_stock_score
from src.ai.ensemble import analyze_stock_ensemble

def test_simple_ai():
    """Test AI system with simple data."""
    print("Testing AI Trading System...")
    
    # Test 1: Pseudo features
    print("\n1. Testing pseudo features...")
    pseudo_data = _build_pseudo_features("TEST")
    print(f"Pseudo features generated: {len(pseudo_data)} categories")
    
    # Test 2: Scoring
    print("\n2. Testing scoring engine...")
    features = pseudo_data['features']
    sentiment_data = pseudo_data['sentiment_data']
    fundamental_data = pseudo_data['fundamental_data']
    
    score = calculate_stock_score("TEST", features, sentiment_data, fundamental_data)
    print(f"Score calculated: {score.overall_score:.3f} (confidence: {score.confidence:.3f})")
    
    # Test 3: Ensemble (without Ollama)
    print("\n3. Testing ensemble analysis...")
    try:
        ensemble_result = analyze_stock_ensemble(
            "TEST", 
            pseudo_data['market_data'], 
            sentiment_data, 
            fundamental_data
        )
        print(f"Ensemble result: {ensemble_result.final_action} (confidence: {ensemble_result.final_confidence:.3f})")
    except Exception as e:
        print(f"Ensemble failed: {e}")
    
    # Test 4: AI Selector with limit
    print("\n4. Testing AI Selector...")
    try:
        agent = AISelectorAgent()
        picks = agent.run(limit=2)
        print(f"AI Selector completed: {len(picks)} picks")
        for pick in picks:
            print(f"   - {pick.symbol}: {pick.action} (score: {pick.score:.3f})")
    except Exception as e:
        print(f"AI Selector failed: {e}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_simple_ai()
