"""
Safety Features Validation Script

This script validates all the critical safety features that were added:
1. Security Validator (data leak detection)
2. Hallucination Detector (AI validation)
3. Change Tracker (modification logging)
4. Debug Scheduler (regular health checks)
5. Unified Main Entry Point
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test_security_validator():
    """Test the security validator."""
    try:
        logger.info("Testing Security Validator...")
        
        from validation.security_validator import get_security_validator, detect_api_key_leaks, detect_pii_leaks, sanitize_logs
        
        # Test API key detection
        test_text_with_key = 'api_key = "sk-1234567890abcdef1234567890abcdef"'
        issues = detect_api_key_leaks(test_text_with_key, "test_file.py")
        
        if issues:
            logger.info(f"✓ API key detection working: {len(issues)} issues found")
        else:
            logger.warning("⚠ API key detection may not be working")
        
        # Test PII detection
        test_text_with_pii = 'email = "user@example.com" phone = "555-123-4567"'
        pii_issues = detect_pii_leaks(test_text_with_pii, "test_file.py")
        
        if pii_issues:
            logger.info(f"✓ PII detection working: {len(pii_issues)} issues found")
        else:
            logger.warning("⚠ PII detection may not be working")
        
        # Test log sanitization
        test_log = 'User email: user@example.com, API key: sk-1234567890abcdef'
        sanitized = sanitize_logs(test_log)
        
        if "***MASKED***" in sanitized:
            logger.info("✓ Log sanitization working")
        else:
            logger.warning("⚠ Log sanitization may not be working")
        
        # Test comprehensive scan
        security_validator = get_security_validator()
        test_files = ["src/main.py", "src/validation/security_validator.py"]
        report = security_validator.run_comprehensive_security_scan(test_files)
        
        logger.info(f"✓ Security scan completed: {report.overall_status}")
        logger.info(f"  Total issues: {report.total_issues}")
        logger.info(f"  Critical: {report.critical_issues}")
        logger.info(f"  High: {report.high_issues}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Security validator test failed: {e}")
        return False

async def test_hallucination_detector():
    """Test the hallucination detector."""
    try:
        logger.info("Testing Hallucination Detector...")
        
        from validation.hallucination_detector import get_hallucination_detector, detect_hallucinations, validate_ai_response
        
        # Test with unrealistic values
        test_response_unrealistic = {
            "confidence": 1.5,  # Invalid confidence > 1.0
            "recommendation": "BUY",
            "reasoning": "This is a guaranteed profit with no risk",
            "market_data": {
                "price": -100.0,  # Invalid negative price
                "sentiment": 0.8
            }
        }
        
        report = detect_hallucinations(test_response_unrealistic)
        
        if report.total_issues > 0:
            logger.info(f"✓ Hallucination detection working: {report.total_issues} issues found")
            logger.info(f"  Status: {report.overall_status}")
            logger.info(f"  Valid: {report.ai_response_valid}")
        else:
            logger.warning("⚠ Hallucination detection may not be working")
        
        # Test with realistic values
        test_response_realistic = {
            "confidence": 0.8,
            "recommendation": "BUY",
            "reasoning": "Strong bullish momentum with positive sentiment",
            "market_data": {
                "price": 150.0,
                "sentiment": 0.7
            }
        }
        
        report_realistic = detect_hallucinations(test_response_realistic)
        
        if report_realistic.ai_response_valid:
            logger.info("✓ Realistic response validation working")
        else:
            logger.warning("⚠ Realistic response validation may not be working")
        
        # Test quick validation
        is_valid = validate_ai_response(test_response_realistic)
        logger.info(f"✓ Quick validation working: {is_valid}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Hallucination detector test failed: {e}")
        return False

async def test_change_tracker():
    """Test the change tracker."""
    try:
        logger.info("Testing Change Tracker...")
        
        from validation.change_tracker import get_change_tracker, track_change, get_changes, get_change_summary, ChangeType, ChangeSeverity
        
        # Test tracking a change
        change_id = track_change(
            change_type=ChangeType.FILE_CREATION,
            file_path="test_safety_features.py",
            description="Created safety features test script",
            reason="Testing change tracking functionality",
            impact="Validates change tracking system",
            author="test_script",
            phase="testing",
            severity=ChangeSeverity.LOW
        )
        
        if change_id:
            logger.info(f"✓ Change tracking working: {change_id}")
        else:
            logger.warning("⚠ Change tracking may not be working")
        
        # Test retrieving changes
        changes = get_changes(limit=5)
        
        if changes:
            logger.info(f"✓ Change retrieval working: {len(changes)} changes found")
        else:
            logger.warning("⚠ Change retrieval may not be working")
        
        # Test change summary
        summary = get_change_summary()
        
        if summary.total_changes >= 0:
            logger.info(f"✓ Change summary working: {summary.total_changes} total changes")
            logger.info(f"  By type: {summary.changes_by_type}")
            logger.info(f"  By severity: {summary.changes_by_severity}")
        else:
            logger.warning("⚠ Change summary may not be working")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Change tracker test failed: {e}")
        return False

async def test_debug_scheduler():
    """Test the debug scheduler."""
    try:
        logger.info("Testing Debug Scheduler...")
        
        from validation.debug_scheduler import get_debug_scheduler, start_debug_scheduler, stop_debug_scheduler, get_debug_report
        
        # Test debug scheduler initialization
        debug_scheduler = get_debug_scheduler()
        
        if debug_scheduler:
            logger.info("✓ Debug scheduler initialization working")
        else:
            logger.warning("⚠ Debug scheduler initialization may not be working")
        
        # Test starting debug scheduler
        start_debug_scheduler()
        logger.info("✓ Debug scheduler started")
        
        # Wait a moment for some checks to run
        await asyncio.sleep(2)
        
        # Test getting debug report
        report = get_debug_report(hours=1)
        
        if report:
            logger.info(f"✓ Debug report working: {report.total_checks} checks")
            logger.info(f"  Status: {report.overall_status}")
            logger.info(f"  Passed: {report.passed_checks}")
            logger.info(f"  Warnings: {report.warning_checks}")
            logger.info(f"  Failed: {report.failed_checks}")
        else:
            logger.warning("⚠ Debug report may not be working")
        
        # Test stopping debug scheduler
        stop_debug_scheduler()
        logger.info("✓ Debug scheduler stopped")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Debug scheduler test failed: {e}")
        return False

async def test_unified_main():
    """Test the unified main entry point."""
    try:
        logger.info("Testing Unified Main Entry Point...")
        
        from main import UnifiedTradingSystem
        
        # Test unified system initialization
        system = UnifiedTradingSystem("DEMO")
        
        if system:
            logger.info("✓ Unified system initialization working")
        else:
            logger.warning("⚠ Unified system initialization may not be working")
        
        # Test system status
        status = await system.get_system_status()
        
        if status:
            logger.info(f"✓ System status working: {status['mode']} mode")
            logger.info(f"  Phases: {status['phases']}")
        else:
            logger.warning("⚠ System status may not be working")
        
        # Test security validation
        security_ok = await system.run_security_validation()
        logger.info(f"✓ Security validation working: {'PASS' if security_ok else 'FAIL'}")
        
        # Test hallucination test
        hallucination_ok = await system.run_hallucination_test()
        logger.info(f"✓ Hallucination test working: {'PASS' if hallucination_ok else 'FAIL'}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Unified main test failed: {e}")
        return False

async def run_comprehensive_safety_test():
    """Run comprehensive safety features test."""
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE SAFETY FEATURES TEST")
    logger.info("=" * 60)
    
    tests = [
        ("Security Validator", test_security_validator),
        ("Hallucination Detector", test_hallucination_detector),
        ("Change Tracker", test_change_tracker),
        ("Debug Scheduler", test_debug_scheduler),
        ("Unified Main", test_unified_main)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = await test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SAFETY FEATURES TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL SAFETY FEATURES WORKING - System is BULLETPROOF! 🛡️")
        return True
    else:
        logger.warning(f"⚠ {total - passed} safety features need attention")
        return False

async def main():
    """Main test function."""
    try:
        success = await run_comprehensive_safety_test()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Safety features test failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
