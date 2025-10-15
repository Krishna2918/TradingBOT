#!/usr/bin/env python3
"""
Final validation script for AI Trading System
"""

import sys
import os
import logging
from datetime import datetime

# Add project root to path
sys.path.append('.')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def validate_ai_system():
    """Validate the AI trading system components."""
    logger.info("Starting AI Trading System validation...")
    
    validation_results = {
        'features': False,
        'factors': False,
        'scoring': False,
        'ensemble': False,
        'ai_selector': False,
        'database': False
    }
    
    try:
        # Test 1: Features module
        logger.info("Testing features module...")
        from src.ai.features import TechnicalIndicators, FeatureEngine
        indicators = TechnicalIndicators()
        engine = FeatureEngine()
        validation_results['features'] = True
        logger.info("Features module: PASSED")
        
    except Exception as e:
        logger.error(f"Features module: FAILED - {e}")
    
    try:
        # Test 2: Factors module
        logger.info("Testing factors module...")
        from src.ai.factors import NewsAPIClient, FinnhubClient, FactorEngine
        factor_engine = FactorEngine()
        validation_results['factors'] = True
        logger.info("Factors module: PASSED")
        
    except Exception as e:
        logger.error(f"Factors module: FAILED - {e}")
    
    try:
        # Test 3: Scoring module
        logger.info("Testing scoring module...")
        from src.ai.scoring import ScoringEngine, calculate_stock_score
        scoring_engine = ScoringEngine()
        validation_results['scoring'] = True
        logger.info("[PASS] Scoring module: PASSED")
        
    except Exception as e:
        logger.error(f"[FAIL] Scoring module: FAILED - {e}")
    
    try:
        # Test 4: Ensemble module
        logger.info("Testing ensemble module...")
        from src.ai.ensemble import OllamaEnsemble, analyze_stock_ensemble
        ensemble = OllamaEnsemble()
        validation_results['ensemble'] = True
        logger.info("[PASS] Ensemble module: PASSED")
        
    except Exception as e:
        logger.error(f"[FAIL] Ensemble module: FAILED - {e}")
    
    try:
        # Test 5: AI Selector
        logger.info("Testing AI Selector...")
        from src.agents.ai_selector import AISelectorAgent
        agent = AISelectorAgent()
        status = agent.get_status()
        validation_results['ai_selector'] = True
        logger.info("[PASS] AI Selector: PASSED")
        
    except Exception as e:
        logger.error(f"[FAIL] AI Selector: FAILED - {e}")
    
    try:
        # Test 6: Database connectivity
        logger.info("Testing database connectivity...")
        import duckdb
        conn = duckdb.connect("data/market_data.duckdb")
        conn.execute("SELECT 1")
        conn.close()
        validation_results['database'] = True
        logger.info("[PASS] Database: PASSED")
        
    except Exception as e:
        logger.error(f"[FAIL] Database: FAILED - {e}")
    
    return validation_results

def run_integration_test():
    """Run integration test with AI Selector."""
    logger.info("Running integration test...")
    
    try:
        from src.agents.ai_selector import AISelectorAgent
        
        agent = AISelectorAgent()
        picks = agent.run(limit=3)
        
        if picks:
            logger.info(f"[PASS] Integration test: PASSED - {len(picks)} picks generated")
            for pick in picks:
                logger.info(f"   - {pick.symbol}: {pick.action} (score: {pick.score:.3f})")
            return True
        else:
            logger.warning("[WARN] Integration test: PARTIAL - No picks generated")
            return False
            
    except Exception as e:
        logger.error(f"[FAIL] Integration test: FAILED - {e}")
        return False

def main():
    """Main validation function."""
    logger.info("=" * 60)
    logger.info("AI TRADING SYSTEM - FINAL VALIDATION")
    logger.info("=" * 60)
    
    # Run component validation
    results = validate_ai_system()
    
    # Run integration test
    integration_success = run_integration_test()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    passed_count = sum(1 for result in results.values() if result)
    total_count = len(results)
    
    for component, passed in results.items():
        status = "[PASS] PASSED" if passed else "[FAIL] FAILED"
        logger.info(f"{component.upper()}: {status}")
    
    logger.info(f"\nINTEGRATION TEST: {'[PASS] PASSED' if integration_success else '[FAIL] FAILED'}")
    
    logger.info(f"\nOVERALL: {passed_count}/{total_count} components passed")
    
    if passed_count == total_count and integration_success:
        logger.info("\n" + "=" * 60)
        logger.info("[PASS][PASS][PASS] ALL TESTS PASSED — SYSTEM READY FOR BUILD [READY]")
        logger.info("=" * 60)
        return True
    else:
        logger.info("\n" + "=" * 60)
        logger.info("[FAIL][FAIL][FAIL] SOME TESTS FAILED — SYSTEM NEEDS ATTENTION")
        logger.info("=" * 60)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)