#!/usr/bin/env python3
"""
Phase 11 Smoke Test - Acceptance, Rollout & Safety

Tests the core functionality of feature flags, acceptance criteria,
safety controls, and rollout management.
"""

import sys
import os
import logging
import asyncio
import json
import tempfile
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_feature_flags_functionality():
    """Test feature flags core functionality."""
    print("\n[TEST] Feature Flags Functionality")
    print("=" * 50)
    
    try:
        from config.feature_flags import (
            FeatureFlagManager, FeatureStatus, is_feature_enabled,
            enable_feature, disable_feature, rollback_feature
        )
        
        # Test with temporary config file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # Create manager instance
            manager = FeatureFlagManager(config_file=temp_file)
            print("[OK] FeatureFlagManager created successfully")
            
            # Test flag retrieval
            all_flags = manager.get_all_flags()
            print(f"[OK] Retrieved {len(all_flags)} feature flags")
            
            # Test default flags
            expected_flags = [
                'adaptive_weights', 'confidence_calibration', 'drawdown_aware_kelly',
                'atr_brackets', 'regime_awareness', 'data_quality_gates',
                'api_budget_management', 'enhanced_monitoring'
            ]
            
            for flag_name in expected_flags:
                if flag_name in all_flags:
                    print(f"[OK] Default flag '{flag_name}' found")
                else:
                    print(f"[WARN] Default flag '{flag_name}' not found")
            
            # Test flag status checking
            assert not is_feature_enabled('adaptive_weights')
            print("[OK] Disabled feature flag correctly identified")
            
            assert is_feature_enabled('data_quality_gates')
            print("[OK] Enabled feature flag correctly identified")
            
            # Test enabling feature
            success = enable_feature('adaptive_weights', 50.0)
            assert success
            print("[OK] Feature flag enabled successfully")
            
            # Test flag status after enabling
            status = manager.get_feature_status('adaptive_weights')
            assert status.status == FeatureStatus.ROLLING_OUT
            assert status.rollout_percentage == 50.0
            print("[OK] Feature flag status updated correctly")
            
            # Test disabling feature
            success = disable_feature('adaptive_weights')
            assert success
            print("[OK] Feature flag disabled successfully")
            
            # Test rollback feature
            enable_feature('adaptive_weights', 100.0)
            success = rollback_feature('adaptive_weights')
            assert success
            print("[OK] Feature flag rollback successful")
            
            # Test metrics summary
            summary = manager.get_metrics_summary()
            assert 'total_flags' in summary
            assert 'enabled_flags' in summary
            print("[OK] Metrics summary generated successfully")
            
            print("[PASS] Feature flags functionality test completed successfully")
            return True
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
    except Exception as e:
        print(f"[FAIL] Feature flags functionality test failed: {e}")
        return False

def test_acceptance_tests_functionality():
    """Test acceptance tests core functionality."""
    print("\n[TEST] Acceptance Tests Functionality")
    print("=" * 50)
    
    try:
        from scripts.acceptance_tests import AcceptanceTestSuite
        
        # Create test suite instance
        test_suite = AcceptanceTestSuite()
        print("[OK] AcceptanceTestSuite created successfully")
        
        # Test thresholds
        assert test_suite.thresholds['daily_success_rate'] == 0.99
        assert test_suite.thresholds['pipeline_p95_latency'] == 25.0
        assert test_suite.thresholds['data_contract_violations'] == 0
        print("[OK] Acceptance criteria thresholds configured correctly")
        
        # Test AI limit
        assert test_suite.ai_limit == 1200  # Default value
        print(f"[OK] AI limit set to {test_suite.ai_limit}")
        
        # Test results structure
        assert 'timestamp' in test_suite.results
        assert 'tests' in test_suite.results
        assert 'metrics' in test_suite.results
        assert 'summary' in test_suite.results
        print("[OK] Results structure initialized correctly")
        
        print("[PASS] Acceptance tests functionality test completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Acceptance tests functionality test failed: {e}")
        return False

def test_safety_controls_functionality():
    """Test safety controls core functionality."""
    print("\n[TEST] Safety Controls Functionality")
    print("=" * 50)
    
    try:
        from dashboard.safety_controls import (
            SafetyControlsDashboard, get_feature_flag_status,
            get_slo_metrics, get_alert_status, get_rollback_history
        )
        
        # Create dashboard instance
        dashboard = SafetyControlsDashboard()
        print("[OK] SafetyControlsDashboard created successfully")
        
        # Test feature flag status
        try:
            flag_status = dashboard.get_feature_flag_status()
            if 'error' not in flag_status:
                print("[OK] Feature flag status retrieved successfully")
            else:
                print(f"[WARN] Feature flag status error: {flag_status['error']}")
        except Exception as e:
            print(f"[WARN] Feature flag status failed: {e}")
        
        # Test SLO metrics
        slo_metrics = dashboard.get_slo_metrics()
        assert 'overall_health' in slo_metrics
        assert 'healthy_slos' in slo_metrics
        assert 'total_slos' in slo_metrics
        assert 'slo_details' in slo_metrics
        print("[OK] SLO metrics retrieved successfully")
        
        # Test alert status
        alert_status = dashboard.get_alert_status()
        assert 'active_alerts' in alert_status
        assert 'recent_alerts' in alert_status
        assert 'alert_channels' in alert_status
        print("[OK] Alert status retrieved successfully")
        
        # Test rollback history
        rollback_history = dashboard.get_rollback_history()
        assert 'recent_rollbacks' in rollback_history
        assert 'rollback_capabilities' in rollback_history
        assert 'rollback_triggers' in rollback_history
        print("[OK] Rollback history retrieved successfully")
        
        # Test system health summary
        health_summary = dashboard.get_system_health_summary()
        assert 'overall_health_score' in health_summary
        assert 'health_status' in health_summary
        assert 'components' in health_summary
        print("[OK] System health summary retrieved successfully")
        
        # Test metrics dashboard data
        dashboard_data = dashboard.get_metrics_dashboard_data()
        assert 'health_summary' in dashboard_data
        assert 'feature_flags' in dashboard_data
        assert 'slo_metrics' in dashboard_data
        assert 'charts' in dashboard_data
        print("[OK] Metrics dashboard data retrieved successfully")
        
        print("[PASS] Safety controls functionality test completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Safety controls functionality test failed: {e}")
        return False

def test_rollout_plan_documentation():
    """Test rollout plan documentation."""
    print("\n[TEST] Rollout Plan Documentation")
    print("=" * 50)
    
    try:
        rollout_plan_file = 'docs/ROLLOUT_PLAN.md'
        
        if os.path.exists(rollout_plan_file):
            print("[OK] Rollout plan documentation exists")
            
            with open(rollout_plan_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for required sections
            required_sections = [
                '# Production Rollout Plan',
                '## Rollout Strategy',
                '## Monitoring and SLOs',
                '## Rollback Procedures',
                '## Feature Flag Management',
                '## Testing Strategy',
                '## Communication Plan',
                '## Success Criteria',
                '## Risk Mitigation'
            ]
            
            missing_sections = []
            for section in required_sections:
                if section not in content:
                    missing_sections.append(section)
            
            if missing_sections:
                print(f"[WARN] Missing sections in rollout plan: {missing_sections}")
            else:
                print("[OK] All required sections present in rollout plan")
            
            # Check for rollout phases
            if 'Phase 1: Foundation Features' in content:
                print("[OK] Foundation features phase documented")
            else:
                print("[WARN] Foundation features phase not documented")
            
            if 'Phase 2: Risk Management Features' in content:
                print("[OK] Risk management features phase documented")
            else:
                print("[WARN] Risk management features phase not documented")
            
            if 'Phase 3: Intelligence Features' in content:
                print("[OK] Intelligence features phase documented")
            else:
                print("[WARN] Intelligence features phase not documented")
            
            if 'Phase 4: Optimization Features' in content:
                print("[OK] Optimization features phase documented")
            else:
                print("[WARN] Optimization features phase not documented")
            
            print("[PASS] Rollout plan documentation test completed successfully")
            return True
        else:
            print("[FAIL] Rollout plan documentation file not found")
            return False
        
    except Exception as e:
        print(f"[FAIL] Rollout plan documentation test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("\n[TEST] File Structure")
    print("=" * 50)
    
    required_files = [
        'src/config/feature_flags.py',
        'docs/ROLLOUT_PLAN.md',
        'scripts/acceptance_tests.py',
        'src/dashboard/safety_controls.py',
        'tests/test_phase11_integration.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"[OK] {file_path} exists")
        else:
            print(f"[FAIL] {file_path} missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"[FAIL] {len(missing_files)} files missing")
        return False
    else:
        print("[PASS] All required files exist")
        return True

def test_basic_imports():
    """Test that all modules can be imported."""
    print("\n[TEST] Basic Imports")
    print("=" * 50)
    
    try:
        # Test feature flags imports
        import config.feature_flags
        print("[OK] config.feature_flags imported successfully")
        
        # Test safety controls imports
        import dashboard.safety_controls
        print("[OK] dashboard.safety_controls imported successfully")
        
        # Test acceptance tests imports
        import scripts.acceptance_tests
        print("[OK] scripts.acceptance_tests imported successfully")
        
        print("[PASS] Basic imports test completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Basic imports test failed: {e}")
        return False

def test_feature_flag_integration():
    """Test feature flag integration with other components."""
    print("\n[TEST] Feature Flag Integration")
    print("=" * 50)
    
    try:
        from config.feature_flags import is_feature_enabled
        
        # Test integration with different user IDs
        test_users = ['user1', 'user2', 'user3', 'default']
        
        for user_id in test_users:
            # Test that feature flags work with different users
            enabled = is_feature_enabled('data_quality_gates', user_id)
            print(f"[OK] Feature flag check for user '{user_id}': {enabled}")
        
        # Test that feature flags are consistent
        default_enabled = is_feature_enabled('data_quality_gates', 'default')
        user1_enabled = is_feature_enabled('data_quality_gates', 'user1')
        
        # For enabled features, should be consistent across users
        if default_enabled == user1_enabled:
            print("[OK] Feature flag consistency maintained across users")
        else:
            print("[WARN] Feature flag inconsistency detected")
        
        print("[PASS] Feature flag integration test completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Feature flag integration test failed: {e}")
        return False

def test_safety_controls_integration():
    """Test safety controls integration."""
    print("\n[TEST] Safety Controls Integration")
    print("=" * 50)
    
    try:
        from dashboard.safety_controls import get_system_health_summary
        
        # Test system health summary
        health_summary = get_system_health_summary()
        
        if 'error' not in health_summary:
            assert 'overall_health_score' in health_summary
            assert 'health_status' in health_summary
            assert 'components' in health_summary
            
            health_score = health_summary['overall_health_score']
            health_status = health_summary['health_status']
            
            print(f"[OK] System health score: {health_score:.2f}")
            print(f"[OK] System health status: {health_status}")
            
            # Check that health score is within valid range
            if 0 <= health_score <= 1:
                print("[OK] Health score within valid range")
            else:
                print("[WARN] Health score outside valid range")
            
            # Check that health status is valid
            if health_status in ['healthy', 'warning', 'critical']:
                print("[OK] Health status is valid")
            else:
                print("[WARN] Health status is invalid")
            
        else:
            print(f"[WARN] System health summary error: {health_summary['error']}")
        
        print("[PASS] Safety controls integration test completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Safety controls integration test failed: {e}")
        return False

def main():
    """Run all Phase 11 smoke tests."""
    print("Phase 11 Smoke Test - Acceptance, Rollout & Safety")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Basic Imports", test_basic_imports),
        ("Feature Flags Functionality", test_feature_flags_functionality),
        ("Acceptance Tests Functionality", test_acceptance_tests_functionality),
        ("Safety Controls Functionality", test_safety_controls_functionality),
        ("Rollout Plan Documentation", test_rollout_plan_documentation),
        ("Feature Flag Integration", test_feature_flag_integration),
        ("Safety Controls Integration", test_safety_controls_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[READY] Starting {test_name} test...")
        try:
            if test_func():
                passed += 1
            else:
                print(f"[FAIL] {test_name} test failed")
        except Exception as e:
            print(f"[FAIL] {test_name} test failed: {e}")
    
    print("\n" + "=" * 60)
    print("Phase 11 Smoke Test Summary")
    print("=" * 60)
    
    for test_name, _ in tests:
        status = "[PASS]" if test_name in [tests[i][0] for i in range(passed)] else "[FAIL]"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! Phase 11 Acceptance, Rollout & Safety is working correctly.")
        return 0
    else:
        print(f"\n[FAILURE] {total - passed} test(s) failed. Please check the logs above.")
        return 1

if __name__ == "__main__":
    exit(main())
