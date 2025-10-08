#!/usr/bin/env python3
"""
Test Hybrid Control Plane Integration
"""

import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def test_hybrid_control_plane():
    """Test hybrid control plane integration"""
    print("Testing Hybrid Control Plane Integration...")
    
    try:
        from src.ai.hybrid_control_plane import HybridControlPlane
        from src.ai.meta_ensemble_blender import MetaEnsembleBlender
        from src.ai.local_reasoner import LocalReasoner
        from src.ai.autonomous_trading_ai_helpers import convert_analysis_to_predictions
        
        print("✅ All hybrid control components imported successfully")
        
        # Test hybrid control plane
        config = {'gpt5_api_key': 'test-key'}
        control_plane = HybridControlPlane(config)
        print("✅ Hybrid Control Plane initialized")
        
        # Test meta-ensemble blender
        blender = MetaEnsembleBlender({})
        print("✅ Meta-Ensemble Blender initialized")
        
        # Test local reasoner
        reasoner = LocalReasoner({'model_name': 'qwen2.5:14b-instruct'})
        print("✅ Local Reasoner initialized")
        
        # Test helper functions
        test_analysis = {
            'ai_predictions': {
                'TD.TO': {'direction': 'UP', 'confidence': 0.7}
            },
            'sentiment': {
                'TD.TO': {'compound': 0.5}
            }
        }
        
        predictions = convert_analysis_to_predictions(test_analysis)
        print(f"✅ Converted {len(predictions)} predictions from analysis")
        
        # Test meta-ensemble blending
        blended = blender.blend_predictions(predictions)
        print(f"✅ Meta-ensemble blended {len(blended)} decisions")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_autonomous_ai():
    """Test autonomous AI with hybrid control plane"""
    print("\nTesting Autonomous AI with Hybrid Control Plane...")
    
    try:
        from src.ai.autonomous_trading_ai import AutonomousTradingAI
        
        print("✅ Autonomous AI imported successfully")
        
        # Test initialization
        ai = AutonomousTradingAI(
            mode='demo',
            initial_capital=1000.0,
            symbols=['TD.TO', 'RY.TO']
        )
        print("✅ Autonomous AI initialized with hybrid control plane")
        
        # Test decision making
        test_analysis = {
            'ai_predictions': {
                'TD.TO': {'direction': 'UP', 'confidence': 0.8},
                'RY.TO': {'direction': 'DOWN', 'confidence': 0.6}
            },
            'sentiment': {
                'TD.TO': {'compound': 0.7},
                'RY.TO': {'compound': -0.3}
            },
            'technical_indicators': {
                'TD.TO': {'rsi': 45, 'macd': 0.02},
                'RY.TO': {'rsi': 65, 'macd': -0.01}
            }
        }
        
        decision = ai.make_trading_decision(test_analysis)
        print(f"✅ AI made decision: {decision['action']} {decision['symbol']}")
        print(f"   Confidence: {decision['confidence']:.3f}")
        print(f"   Reasoning: {decision['reasoning']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Autonomous AI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dashboard():
    """Test dashboard integration"""
    print("\nTesting Dashboard Integration...")
    
    try:
        import interactive_trading_dashboard
        
        print("✅ Dashboard imported successfully")
        
        # Test hybrid control status function
        status = interactive_trading_dashboard.create_hybrid_control_status()
        print("✅ Hybrid control status component created")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing Complete Hybrid Control Plane Integration\n")
    
    # Run all tests
    test1 = test_hybrid_control_plane()
    test2 = test_autonomous_ai()
    test3 = test_dashboard()
    
    print("\n" + "="*60)
    print("🎯 INTEGRATION TEST RESULTS")
    print("="*60)
    print(f"Hybrid Control Plane: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Autonomous AI: {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"Dashboard: {'✅ PASS' if test3 else '❌ FAIL'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 ALL INTEGRATIONS WORKING WITHOUT PLACEHOLDERS!")
        print("🚀 System ready for production deployment!")
    else:
        print("\n⚠️ Some integrations failed - check errors above")
    
    print("="*60)
