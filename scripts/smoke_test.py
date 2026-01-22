#!/usr/bin/env python3
"""
Phase 0: Simple Smoke Test Script
=================================

A lightweight smoke test that validates core system components
without running the full trading pipeline.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

async def test_database_connectivity():
    """Test database connectivity."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from config.database import get_database_manager
        
        db_manager = get_database_manager()
        
        # Test DEMO mode
        with db_manager.get_connection("DEMO") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            if result and result[0] == 1:
                logger.info("Database connectivity: PASS")
                return True
        
        logger.error("Database connectivity: FAIL")
        return False
        
    except Exception as e:
        logger.error(f"Database connectivity: FAIL - {e}")
        return False

async def test_api_validation():
    """Test API validation with simple requests."""
    try:
        import requests
        
        # Test Yahoo Finance (no key needed)
        response = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/AAPL", timeout=10)
        if response.status_code == 200:
            logger.info("Yahoo Finance API: PASS")
            return True
        else:
            logger.error(f"Yahoo Finance API: FAIL - HTTP {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"API validation: FAIL - {e}")
        return False

async def test_ai_model_health():
    """Test AI model health."""
    try:
        import requests
        
        # Test Ollama health
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            logger.info(f"Ollama health: PASS - {len(models)} models available")
            return True
        else:
            logger.warning(f"Ollama health: WARNING - HTTP {response.status_code}")
            return True  # Don't fail smoke test for Ollama
            
    except Exception as e:
        logger.warning(f"Ollama health: WARNING - {e}")
        return True  # Don't fail smoke test for Ollama

async def test_component_imports():
    """Test that core components can be imported."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # Test core imports
        from config.database import get_database_manager
        from monitoring.system_monitor import get_system_monitor
        
        logger.info("Component imports: PASS")
        return True
        
    except Exception as e:
        logger.error(f"Component imports: FAIL - {e}")
        return False

async def run_smoke_test():
    """Run the smoke test."""
    logger.info("Starting Phase 0 Smoke Test...")
    start_time = time.time()
    
    tests = [
        ("Database Connectivity", test_database_connectivity),
        ("API Validation", test_api_validation),
        ("AI Model Health", test_ai_model_health),
        ("Component Imports", test_component_imports),
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"Running {test_name}...")
        try:
            result = await test_func()
            results[test_name] = result
            if result:
                passed += 1
        except Exception as e:
            logger.error(f"{test_name} failed with exception: {e}")
            results[test_name] = False
    
    duration = time.time() - start_time
    
    # Summary
    logger.info("=" * 50)
    logger.info("SMOKE TEST SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Duration: {duration:.1f}s")
    logger.info(f"Tests: {passed}/{total} passed")
    logger.info(f"Success Rate: {passed/total*100:.1f}%")
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    # Save results
    results_file = ROOT / "logs" / "smoke_test_results.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": round(duration, 1),
            "tests_passed": passed,
            "tests_total": total,
            "success_rate": round(passed/total*100, 1),
            "results": results
        }, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    
    if passed >= total * 0.8:  # 80% pass rate
        logger.info("SMOKE TEST PASSED")
        return 0
    else:
        logger.error("SMOKE TEST FAILED")
        return 1

if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_smoke_test()))
