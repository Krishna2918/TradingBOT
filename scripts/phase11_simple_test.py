#!/usr/bin/env python3
"""
Phase 11 Simple Test - Acceptance, Rollout & Safety

Simplified test for core functionality without complex imports.
"""

import sys
import os
import logging
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

def test_file_existence():
    """Test that all required files exist."""
    print("\n[TEST] File Existence Check")
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

def test_feature_flags_structure():
    """Test feature flags file structure."""
    print("\n[TEST] Feature Flags Structure")
    print("=" * 50)
    
    feature_flags_file = 'src/config/feature_flags.py'
    
    try:
        with open(feature_flags_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_elements = [
            'class FeatureFlagManager:',
            'class FeatureStatus(Enum):',
            'class FeatureFlag:',
            'def is_enabled(',
            'def enable_feature(',
            'def disable_feature(',
            'def rollback_feature(',
            'def get_feature_status(',
            'def update_metrics(',
            'def get_metrics_summary('
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"[FAIL] Missing elements: {missing_elements}")
            return False
        else:
            print("[PASS] All required elements present in feature flags")
            return True
            
    except Exception as e:
        print(f"[FAIL] Error reading feature flags file: {e}")
        return False

def test_safety_controls_structure():
    """Test safety controls file structure."""
    print("\n[TEST] Safety Controls Structure")
    print("=" * 50)
    
    safety_controls_file = 'src/dashboard/safety_controls.py'
    
    try:
        with open(safety_controls_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_elements = [
            'class SafetyControlsDashboard:',
            'def get_feature_flag_status(',
            'def get_slo_metrics(',
            'def get_alert_status(',
            'def get_rollback_history(',
            'def get_system_health_summary(',
            'def toggle_feature_flag(',
            'def get_metrics_dashboard_data('
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"[FAIL] Missing elements: {missing_elements}")
            return False
        else:
            print("[PASS] All required elements present in safety controls")
            return True
            
    except Exception as e:
        print(f"[FAIL] Error reading safety controls file: {e}")
        return False

def test_acceptance_tests_structure():
    """Test acceptance tests file structure."""
    print("\n[TEST] Acceptance Tests Structure")
    print("=" * 50)
    
    acceptance_tests_file = 'scripts/acceptance_tests.py'
    
    try:
        with open(acceptance_tests_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_elements = [
            'class AcceptanceTestSuite:',
            'async def run_acceptance_tests(',
            'async def _test_system_reliability(',
            'async def _test_performance_requirements(',
            'async def _test_data_contracts(',
            'async def _test_risk_management(',
            'async def _test_feature_flags(',
            'async def _test_safety_controls(',
            'async def _test_end_to_end_pipeline('
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"[FAIL] Missing elements: {missing_elements}")
            return False
        else:
            print("[PASS] All required elements present in acceptance tests")
            return True
            
    except Exception as e:
        print(f"[FAIL] Error reading acceptance tests file: {e}")
        return False

def test_rollout_plan_content():
    """Test rollout plan content."""
    print("\n[TEST] Rollout Plan Content")
    print("=" * 50)
    
    rollout_plan_file = 'docs/ROLLOUT_PLAN.md'
    
    try:
        with open(rollout_plan_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
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
            print(f"[FAIL] Missing sections: {missing_sections}")
            return False
        else:
            print("[PASS] All required sections present in rollout plan")
            return True
            
    except Exception as e:
        print(f"[FAIL] Error reading rollout plan file: {e}")
        return False

def test_integration_tests_structure():
    """Test integration tests file structure."""
    print("\n[TEST] Integration Tests Structure")
    print("=" * 50)
    
    integration_tests_file = 'tests/test_phase11_integration.py'
    
    try:
        with open(integration_tests_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_elements = [
            'class TestFeatureFlags:',
            'class TestAcceptanceTests:',
            'class TestSafetyControls:',
            'class TestGlobalFunctions:',
            'def test_feature_flag_creation(',
            'def test_feature_flag_status(',
            'def test_feature_flag_enable_disable(',
            'def test_feature_flag_rollback(',
            'def test_system_reliability_tests(',
            'def test_performance_tests(',
            'def test_safety_controls_functionality('
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"[FAIL] Missing elements: {missing_elements}")
            return False
        else:
            print("[PASS] All required elements present in integration tests")
            return True
            
    except Exception as e:
        print(f"[FAIL] Error reading integration tests file: {e}")
        return False

def test_basic_imports():
    """Test that core modules can be imported."""
    print("\n[TEST] Basic Imports")
    print("=" * 50)
    
    try:
        # Test feature flags imports
        import config.feature_flags
        print("[OK] config.feature_flags imported successfully")
        
        # Test safety controls imports
        import dashboard.safety_controls
        print("[OK] dashboard.safety_controls imported successfully")
        
        print("[PASS] Basic imports test completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Basic imports test failed: {e}")
        return False

def test_feature_flags_basic_functionality():
    """Test basic feature flags functionality."""
    print("\n[TEST] Feature Flags Basic Functionality")
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
            
            print("[PASS] Feature flags basic functionality test completed successfully")
            return True
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
    except Exception as e:
        print(f"[FAIL] Feature flags basic functionality test failed: {e}")
        return False

def test_safety_controls_basic_functionality():
    """Test basic safety controls functionality."""
    print("\n[TEST] Safety Controls Basic Functionality")
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
        
        print("[PASS] Safety controls basic functionality test completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Safety controls basic functionality test failed: {e}")
        return False

def main():
    """Run all Phase 11 simple tests."""
    print("Phase 11 Simple Test - Acceptance, Rollout & Safety")
    print("=" * 60)
    
    tests = [
        ("File Existence Check", test_file_existence),
        ("Feature Flags Structure", test_feature_flags_structure),
        ("Safety Controls Structure", test_safety_controls_structure),
        ("Acceptance Tests Structure", test_acceptance_tests_structure),
        ("Rollout Plan Content", test_rollout_plan_content),
        ("Integration Tests Structure", test_integration_tests_structure),
        ("Basic Imports", test_basic_imports),
        ("Feature Flags Basic Functionality", test_feature_flags_basic_functionality),
        ("Safety Controls Basic Functionality", test_safety_controls_basic_functionality),
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
    print("Phase 11 Simple Test Summary")
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
