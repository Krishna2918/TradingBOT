"""
Test script for Local LLM Integration
Tests the Qwen2.5-14B-Instruct integration with trading system
"""

import sys
import os
import logging
from datetime import datetime
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_local_llm_client():
    """Test basic LocalLLMClient functionality"""
    try:
        from src.ai.local_llm_integration import LocalLLMClient, LLMRequest
        
        logger.info("Testing LocalLLMClient...")
        
        # Create client with default config
        config = {
            'ollama_url': 'http://localhost:11434',
            'model_name': 'qwen2.5:14b-instruct',
            'timeout': 30
        }
        
        client = LocalLLMClient(config)
        
        # Test health check
        logger.info("Testing health check...")
        health = client.health_check()
        logger.info(f"Health check result: {health}")
        
        if health['status'] != 'healthy':
            logger.error("Ollama service is not healthy")
            return False
        
        # Test basic request
        logger.info("Testing basic LLM request...")
        request = LLMRequest(
            prompt="Hello, can you help me analyze a stock trade?",
            task_type="test",
            max_tokens=100
        )
        
        response = client._make_request(request)
        logger.info(f"LLM Response: {response.content[:200]}...")
        logger.info(f"Response time: {response.response_time:.2f}s")
        logger.info(f"Tokens used: {response.tokens_used}")
        
        # Test performance metrics
        metrics = client.get_performance_metrics()
        logger.info(f"Performance metrics: {metrics}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing LocalLLMClient: {e}")
        return False

def test_trade_narrative_generator():
    """Test TradeNarrativeGenerator with local LLM"""
    try:
        from src.analytics.trade_narrative_generator import (
            TradeNarrativeGenerator, TradeContext, NarrativeType, NarrativeTone
        )
        
        logger.info("Testing TradeNarrativeGenerator with local LLM...")
        
        # Create generator with LLM config
        config = {
            'use_local_llm': True,
            'local_llm_config': {
                'ollama_url': 'http://localhost:11434',
                'model_name': 'qwen2.5:14b-instruct'
            }
        }
        
        generator = TradeNarrativeGenerator(config)
        
        # Create test trade context
        trade_context = TradeContext(
            symbol="RY.TO",
            action="BUY",
            quantity=100,
            price=113.26,
            timestamp=datetime.now(),
            market_data={
                'technical_signal': 'Strong bullish momentum',
                'market_condition': 'Bull market',
                'volatility_level': 'Low',
                'market_regime': 'BULL'
            },
            ai_analysis={
                'confidence': 85,
                'risk_level': 'Low',
                'key_factors': 'Strong fundamentals and technical breakout'
            },
            risk_metrics={
                'var_95': 0.025,
                'beta': 0.8,
                'max_drawdown': 0.03,
                'risk_level': 'Low'
            },
            performance_metrics={
                'pnl_pct': 2.5,
                'pnl_amount': 283.15,
                'sharpe_ratio': 1.8,
                'total_return': 12.5
            }
        )
        
        # Test narrative generation
        logger.info("Generating trade decision narrative...")
        narrative = generator.generate_trade_narrative(
            trade_context, 
            NarrativeType.TRADE_DECISION, 
            NarrativeTone.PROFESSIONAL,
            use_llm=True
        )
        
        logger.info(f"Generated narrative: {narrative}")
        
        # Test statistics
        stats = generator.get_narrative_statistics()
        logger.info(f"Narrative statistics: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing TradeNarrativeGenerator: {e}")
        return False

def test_feature_conflict_detector():
    """Test FeatureConflictDetector with local LLM"""
    try:
        from src.ai.feature_conflict_detector import (
            FeatureConflictDetector, FeatureDefinition, ConflictType
        )
        
        logger.info("Testing FeatureConflictDetector with local LLM...")
        
        # Create detector with LLM config
        config = {
            'use_local_llm': True,
            'local_llm_config': {
                'ollama_url': 'http://localhost:11434',
                'model_name': 'qwen2.5:14b-instruct'
            },
            'correlation_threshold': 0.7,
            'redundancy_threshold': 0.9
        }
        
        detector = FeatureConflictDetector(config)
        
        # Register test features
        features = [
            FeatureDefinition(
                name="RSI_14",
                description="14-period Relative Strength Index",
                category="momentum",
                calculation_method="RSI calculation over 14 periods",
                parameters={'period': 14},
                dependencies=['close_price'],
                expected_range=(0, 100),
                interpretation="Values above 70 indicate overbought, below 30 oversold"
            ),
            FeatureDefinition(
                name="RSI_21",
                description="21-period Relative Strength Index",
                category="momentum",
                calculation_method="RSI calculation over 21 periods",
                parameters={'period': 21},
                dependencies=['close_price'],
                expected_range=(0, 100),
                interpretation="Values above 70 indicate overbought, below 30 oversold"
            ),
            FeatureDefinition(
                name="MACD",
                description="Moving Average Convergence Divergence",
                category="trend",
                calculation_method="EMA(12) - EMA(26)",
                parameters={'fast': 12, 'slow': 26},
                dependencies=['close_price'],
                expected_range=(-10, 10),
                interpretation="Positive values indicate bullish momentum"
            ),
            FeatureDefinition(
                name="BOLLINGER_UPPER",
                description="Bollinger Bands Upper Band",
                category="volatility",
                calculation_method="SMA(20) + 2*STD(20)",
                parameters={'period': 20, 'std_dev': 2},
                dependencies=['close_price'],
                expected_range=(50, 200),
                interpretation="Price above upper band indicates overbought"
            )
        ]
        
        detector.register_features_batch(features)
        
        # Test conflict detection
        logger.info("Detecting feature conflicts...")
        conflicts = detector.detect_conflicts()
        
        logger.info(f"Detected {len(conflicts)} conflicts:")
        for conflict in conflicts:
            logger.info(f"  - {conflict.conflict_type.value}: {conflict.description}")
            logger.info(f"    Severity: {conflict.severity.value}")
            logger.info(f"    Features: {conflict.features_involved}")
            logger.info(f"    Recommendation: {conflict.recommendation}")
        
        # Test summary
        summary = detector.get_conflict_summary()
        logger.info(f"Conflict summary: {summary}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing FeatureConflictDetector: {e}")
        return False

def test_advanced_llm_features():
    """Test advanced LLM features"""
    try:
        from src.ai.local_llm_integration import LocalLLMClient, LLMRequest
        
        logger.info("Testing advanced LLM features...")
        
        config = {
            'ollama_url': 'http://localhost:11434',
            'model_name': 'qwen2.5:14b-instruct'
        }
        
        client = LocalLLMClient(config)
        
        # Test market regime analysis
        logger.info("Testing market regime analysis...")
        market_data = {
            'volatility': 0.15,
            'trend': 'bullish',
            'volume': 'high',
            'sector_performance': {'technology': 0.05, 'finance': 0.02},
            'economic_indicators': {'gdp_growth': 0.03, 'inflation': 0.025},
            'news_sentiment': 'positive'
        }
        
        response = client.analyze_market_regime(market_data)
        logger.info(f"Market regime analysis: {response.content[:300]}...")
        
        # Test risk assessment
        logger.info("Testing risk assessment...")
        portfolio_data = {
            'total_value': 100000,
            'positions': {'RY.TO': 50000, 'TD.TO': 30000, 'SHOP.TO': 20000},
            'sector_allocation': {'finance': 0.8, 'technology': 0.2},
            'var_95': 0.025,
            'max_drawdown': 0.05,
            'beta': 0.9,
            'correlation': 0.3
        }
        
        response = client.generate_risk_assessment(portfolio_data)
        logger.info(f"Risk assessment: {response.content[:300]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing advanced LLM features: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting Local LLM Integration Tests")
    logger.info("=" * 50)
    
    tests = [
        ("LocalLLMClient Basic Test", test_local_llm_client),
        ("TradeNarrativeGenerator Test", test_trade_narrative_generator),
        ("FeatureConflictDetector Test", test_feature_conflict_detector),
        ("Advanced LLM Features Test", test_advanced_llm_features)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name}...")
        logger.info("-" * 30)
        
        try:
            success = test_func()
            results[test_name] = "PASSED" if success else "FAILED"
            logger.info(f"{test_name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            results[test_name] = f"ERROR: {str(e)}"
            logger.error(f"{test_name}: ERROR - {e}")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    for test_name, result in results.items():
        logger.info(f"{test_name}: {result}")
    
    passed = sum(1 for result in results.values() if result == "PASSED")
    total = len(results)
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("All tests passed! Local LLM integration is working correctly.")
        return True
    else:
        logger.warning(f"{total - passed} tests failed. Please check the logs for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
