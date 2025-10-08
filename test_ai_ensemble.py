"""
Test AI Ensemble Integration
Verify Grok, Kimi, and Claude are working together
"""

import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from src.ai.ai_ensemble import AIEnsemble

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ai_ensemble():
    """Test the AI ensemble with mock data"""
    
    print("=" * 70)
    print("🤖 TESTING AI ENSEMBLE INTEGRATION")
    print("=" * 70)
    
    try:
        # Initialize AI ensemble
        print("🔄 Initializing AI ensemble...")
        ensemble = AIEnsemble()
        print("✅ AI ensemble initialized successfully")
        
        # Mock market data
        market_data = {
            'RY.TO': 132.50,
            'TD.TO': 88.75,
            'SHOP.TO': 102.30,
            'CNR.TO': 165.80,
            'ENB.TO': 54.20
        }
        
        # Mock positions
        positions = {
            'RY.TO': {'quantity': 50, 'avg_price': 130.00}
        }
        
        print(f"\n📊 Market Data: {len(market_data)} Canadian stocks")
        print(f"💼 Current Positions: {len(positions)}")
        
        # Get AI analysis
        print("\n🧠 Getting ensemble analysis...")
        analysis = ensemble.analyze_market(market_data, positions)
        
        # Display results
        print("\n" + "=" * 70)
        print("📈 AI ENSEMBLE ANALYSIS RESULTS")
        print("=" * 70)
        
        ensemble_decision = analysis['ensemble_decision']
        print(f"🎯 Final Signal: {ensemble_decision['signal'].upper()}")
        print(f"📊 Confidence: {ensemble_decision['confidence']:.2f}")
        print(f"💭 Reasoning: {ensemble_decision['reason']}")
        print(f"🤝 Consensus: {ensemble_decision.get('consensus', 'unknown')}")
        
        print(f"\n🤖 Individual AI Signals:")
        for ai_name, signal in ensemble_decision.get('individual_signals', {}).items():
            print(f"   • {ai_name.capitalize()}: {signal.upper()}")
        
        print(f"\n📋 Individual AI Analyses:")
        for ai_name, ai_analysis in analysis['individual_analyses'].items():
            if 'error' not in ai_analysis:
                print(f"\n   {ai_name.capitalize()}:")
                print(f"     Signal: {ai_analysis.get('signal', 'N/A').upper()}")
                print(f"     Confidence: {ai_analysis.get('confidence', 0):.2f}")
                print(f"     Reason: {ai_analysis.get('reason', 'N/A')[:100]}...")
            else:
                print(f"\n   {ai_name.capitalize()}: ❌ {ai_analysis['error']}")
        
        print("\n" + "=" * 70)
        print("✅ AI Ensemble test completed successfully!")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ AI Ensemble test failed: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check API keys in config/ai_ensemble_config.yaml")
        print("   2. Verify internet connection")
        print("   3. Ensure API accounts are active")
        print("   4. Check rate limits")
        return False

if __name__ == "__main__":
    success = test_ai_ensemble()
    
    if success:
        print("\n🎉 AI Ensemble is ready for trading!")
        print("🚀 Start demo with: python src/dashboard/demo_dashboard.py")
    else:
        print("\n⚠️  AI Ensemble needs configuration")
        print("📖 See: AI_ENSEMBLE_SETUP.md for setup instructions")
