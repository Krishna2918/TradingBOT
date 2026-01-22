"""
Phase 11 Integration Tests - Acceptance, Rollout & Safety

Tests the feature flags, acceptance criteria, safety controls, and rollout functionality.
"""

import pytest
import asyncio
import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestFeatureFlags:
    """Test feature flag functionality."""
    
    @pytest.fixture
    def feature_flag_manager(self):
        """Create a feature flag manager instance for testing."""
        from config.feature_flags import FeatureFlagManager, FeatureStatus
        
        # Use temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            manager = FeatureFlagManager(config_file=temp_file)
            yield manager
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_feature_flag_creation(self, feature_flag_manager):
        """Test feature flag creation and initialization."""
        flags = feature_flag_manager.get_all_flags()
        
        assert len(flags) > 0
        assert 'adaptive_weights' in flags
        assert 'confidence_calibration' in flags
        assert 'drawdown_aware_kelly' in flags
        assert 'atr_brackets' in flags
        assert 'regime_awareness' in flags
    
    def test_feature_flag_status(self, feature_flag_manager):
        """Test feature flag status checking."""
        # Test disabled feature
        assert not feature_flag_manager.is_enabled('adaptive_weights')
        
        # Test enabled feature
        assert feature_flag_manager.is_enabled('data_quality_gates')
        
        # Test non-existent feature
        assert not feature_flag_manager.is_enabled('non_existent_feature')
    
    def test_feature_flag_enable_disable(self, feature_flag_manager):
        """Test enabling and disabling feature flags."""
        # Enable feature
        success = feature_flag_manager.enable_feature('adaptive_weights', 50.0)
        assert success
        assert feature_flag_manager.is_enabled('adaptive_weights')
        
        # Check status
        status = feature_flag_manager.get_feature_status('adaptive_weights')
        assert status.status == FeatureStatus.ROLLING_OUT
        assert status.rollout_percentage == 50.0
        
        # Disable feature
        success = feature_flag_manager.disable_feature('adaptive_weights')
        assert success
        assert not feature_flag_manager.is_enabled('adaptive_weights')
        
        # Check status
        status = feature_flag_manager.get_feature_status('adaptive_weights')
        assert status.status == FeatureStatus.DISABLED
    
    def test_feature_flag_rollback(self, feature_flag_manager):
        """Test feature flag rollback."""
        # Enable feature first
        feature_flag_manager.enable_feature('adaptive_weights', 100.0)
        assert feature_flag_manager.is_enabled('adaptive_weights')
        
        # Rollback feature
        success = feature_flag_manager.rollback_feature('adaptive_weights')
        assert success
        assert not feature_flag_manager.is_enabled('adaptive_weights')
        
        # Check status
        status = feature_flag_manager.get_feature_status('adaptive_weights')
        assert status.status == FeatureStatus.ROLLBACK
    
    def test_user_specific_flags(self, feature_flag_manager):
        """Test user-specific feature flag behavior."""
        # Enable feature for specific user
        flag = feature_flag_manager.get_feature_status('adaptive_weights')
        flag.enabled_for_users = ['user1', 'user2']
        flag.disabled_for_users = ['user3']
        
        assert feature_flag_manager.is_enabled('adaptive_weights', 'user1')
        assert feature_flag_manager.is_enabled('adaptive_weights', 'user2')
        assert not feature_flag_manager.is_enabled('adaptive_weights', 'user3')
        assert not feature_flag_manager.is_enabled('adaptive_weights', 'user4')
    
    def test_rollout_percentage(self, feature_flag_manager):
        """Test rollout percentage functionality."""
        # Enable feature with 50% rollout
        feature_flag_manager.enable_feature('adaptive_weights', 50.0)
        
        # Test multiple users (some should be enabled, some not)
        enabled_count = 0
        for i in range(100):
            if feature_flag_manager.is_enabled('adaptive_weights', f'user{i}'):
                enabled_count += 1
        
        # Should be approximately 50% (allowing for hash distribution variance)
        assert 30 <= enabled_count <= 70
    
    def test_metrics_update(self, feature_flag_manager):
        """Test metrics update functionality."""
        # Update metrics for a feature
        metrics = {
            'accuracy': 0.85,
            'brier_score': 0.25,
            'trades_count': 100
        }
        
        feature_flag_manager.update_metrics('adaptive_weights', metrics)
        
        # Check metrics were saved
        assert 'adaptive_weights' in feature_flag_manager.metrics
        assert feature_flag_manager.metrics['adaptive_weights']['accuracy'] == 0.85
        assert feature_flag_manager.metrics['adaptive_weights']['brier_score'] == 0.25
    
    def test_metrics_threshold_check(self, feature_flag_manager):
        """Test metrics threshold checking."""
        # Set up feature with threshold
        flag = feature_flag_manager.get_feature_status('adaptive_weights')
        flag.metrics_threshold = {
            'brier_score_threshold': 0.3,
            'accuracy_threshold': 0.6
        }
        
        # Update metrics within threshold
        feature_flag_manager.update_metrics('adaptive_weights', {
            'brier_score': 0.25,
            'accuracy': 0.7
        })
        
        # Should not trigger rollback
        assert flag.status != FeatureStatus.ROLLBACK
        
        # Update metrics exceeding threshold
        feature_flag_manager.update_metrics('adaptive_weights', {
            'brier_score': 0.35,  # Exceeds threshold
            'accuracy': 0.5
        })
        
        # Should trigger rollback if auto-rollback is enabled
        if os.environ.get('AUTO_ROLLBACK_ON_THRESHOLD', 'false').lower() == 'true':
            assert flag.status == FeatureStatus.ROLLBACK
    
    def test_metrics_summary(self, feature_flag_manager):
        """Test metrics summary generation."""
        summary = feature_flag_manager.get_metrics_summary()
        
        assert 'total_flags' in summary
        assert 'enabled_flags' in summary
        assert 'rolling_out_flags' in summary
        assert 'disabled_flags' in summary
        assert 'rollback_flags' in summary
        assert 'flags_with_metrics' in summary
        assert 'last_updated' in summary
        
        assert summary['total_flags'] > 0
        assert summary['enabled_flags'] >= 0
        assert summary['rolling_out_flags'] >= 0
        assert summary['disabled_flags'] >= 0
        assert summary['rollback_flags'] >= 0

class TestAcceptanceTests:
    """Test acceptance test functionality."""
    
    @pytest.fixture
    def acceptance_test_suite(self):
        """Create an acceptance test suite instance for testing."""
        from scripts.acceptance_tests import AcceptanceTestSuite
        return AcceptanceTestSuite()
    
    @pytest.mark.asyncio
    async def test_system_reliability_tests(self, acceptance_test_suite):
        """Test system reliability test functionality."""
        await acceptance_test_suite._test_system_reliability()
        
        assert 'tests' in acceptance_test_suite.results
        assert 'system_reliability' in acceptance_test_suite.results['tests']
        
        reliability_tests = acceptance_test_suite.results['tests']['system_reliability']
        assert 'uptime' in reliability_tests
        assert 'system_health' in reliability_tests
        assert 'database_connectivity' in reliability_tests
        assert 'ai_availability' in reliability_tests
    
    @pytest.mark.asyncio
    async def test_performance_tests(self, acceptance_test_suite):
        """Test performance test functionality."""
        await acceptance_test_suite._test_performance_requirements()
        
        assert 'tests' in acceptance_test_suite.results
        assert 'performance' in acceptance_test_suite.results['tests']
        
        performance_tests = acceptance_test_suite.results['tests']['performance']
        assert 'pipeline_latency' in performance_tests
        assert 'decision_latency' in performance_tests
        assert 'data_freshness' in performance_tests
        assert 'throughput' in performance_tests
    
    @pytest.mark.asyncio
    async def test_data_contract_tests(self, acceptance_test_suite):
        """Test data contract test functionality."""
        await acceptance_test_suite._test_data_contracts()
        
        assert 'tests' in acceptance_test_suite.results
        assert 'data_contracts' in acceptance_test_suite.results['tests']
        
        contract_tests = acceptance_test_suite.results['tests']['data_contracts']
        assert 'data_quality' in contract_tests
        assert 'data_completeness' in contract_tests
        assert 'data_consistency' in contract_tests
    
    @pytest.mark.asyncio
    async def test_risk_management_tests(self, acceptance_test_suite):
        """Test risk management test functionality."""
        await acceptance_test_suite._test_risk_management()
        
        assert 'tests' in acceptance_test_suite.results
        assert 'risk_management' in acceptance_test_suite.results['tests']
        
        risk_tests = acceptance_test_suite.results['tests']['risk_management']
        assert 'kelly_cap' in risk_tests
        assert 'sl_tp_presence' in risk_tests
        assert 'position_sizing' in risk_tests
        assert 'drawdown_management' in risk_tests
    
    @pytest.mark.asyncio
    async def test_feature_flag_tests(self, acceptance_test_suite):
        """Test feature flag test functionality."""
        await acceptance_test_suite._test_feature_flags()
        
        assert 'tests' in acceptance_test_suite.results
        assert 'feature_flags' in acceptance_test_suite.results['tests']
        
        flag_tests = acceptance_test_suite.results['tests']['feature_flags']
        assert 'flag_retrieval' in flag_tests
        assert 'flag_status' in flag_tests
        assert 'metrics_summary' in flag_tests
    
    @pytest.mark.asyncio
    async def test_safety_controls_tests(self, acceptance_test_suite):
        """Test safety controls test functionality."""
        await acceptance_test_suite._test_safety_controls()
        
        assert 'tests' in acceptance_test_suite.results
        assert 'safety_controls' in acceptance_test_suite.results['tests']
        
        safety_tests = acceptance_test_suite.results['tests']['safety_controls']
        assert 'monitoring' in safety_tests
        assert 'alerting' in safety_tests
        assert 'rollback' in safety_tests
        assert 'circuit_breakers' in safety_tests
    
    @pytest.mark.asyncio
    async def test_end_to_end_tests(self, acceptance_test_suite):
        """Test end-to-end test functionality."""
        await acceptance_test_suite._test_end_to_end_pipeline()
        
        assert 'tests' in acceptance_test_suite.results
        assert 'end_to_end' in acceptance_test_suite.results['tests']
        
        e2e_tests = acceptance_test_suite.results['tests']['end_to_end']
        assert 'trading_cycle' in e2e_tests
        assert 'data_flow' in e2e_tests
        assert 'decision_making' in e2e_tests
        assert 'execution' in e2e_tests
    
    def test_summary_generation(self, acceptance_test_suite):
        """Test summary generation."""
        # Set up some test results
        acceptance_test_suite.results['tests'] = {
            'system_reliability': {
                'uptime': {'passed': True},
                'system_health': {'passed': True}
            },
            'performance': {
                'pipeline_latency': {'passed': True},
                'decision_latency': {'passed': True}
            }
        }
        
        acceptance_test_suite._generate_summary()
        
        assert 'summary' in acceptance_test_suite.results
        summary = acceptance_test_suite.results['summary']
        
        assert 'total_time_seconds' in summary
        assert 'ai_limit' in summary
        assert 'status' in summary
        assert 'criteria_met' in summary
        assert 'total_criteria' in summary
        assert 'criteria_percentage' in summary

class TestSafetyControls:
    """Test safety controls functionality."""
    
    @pytest.fixture
    def safety_dashboard(self):
        """Create a safety controls dashboard instance for testing."""
        from dashboard.safety_controls import SafetyControlsDashboard
        return SafetyControlsDashboard()
    
    def test_feature_flag_status(self, safety_dashboard):
        """Test feature flag status retrieval."""
        with patch('config.feature_flags.get_feature_flag_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.get_all_flags.return_value = {
                'test_flag': Mock(
                    name='test_flag',
                    status=Mock(value='enabled'),
                    description='Test flag',
                    rollout_percentage=100.0,
                    dependencies=[],
                    metrics_threshold={},
                    updated_at=datetime.now()
                )
            }
            mock_manager.get_metrics_summary.return_value = {
                'total_flags': 1,
                'enabled_flags': 1,
                'rolling_out_flags': 0,
                'disabled_flags': 0,
                'rollback_flags': 0,
                'flags_with_metrics': 0
            }
            mock_manager.is_enabled.return_value = True
            mock_get_manager.return_value = mock_manager
            
            status = safety_dashboard.get_feature_flag_status()
            
            assert 'flags' in status
            assert 'summary' in status
            assert 'last_updated' in status
            assert 'test_flag' in status['flags']
    
    def test_slo_metrics(self, safety_dashboard):
        """Test SLO metrics retrieval."""
        metrics = safety_dashboard.get_slo_metrics()
        
        assert 'overall_health' in metrics
        assert 'healthy_slos' in metrics
        assert 'total_slos' in metrics
        assert 'slo_details' in metrics
        assert 'last_updated' in metrics
        
        assert metrics['total_slos'] > 0
        assert metrics['healthy_slos'] >= 0
        assert 0 <= metrics['overall_health'] <= 1
    
    def test_alert_status(self, safety_dashboard):
        """Test alert status retrieval."""
        status = safety_dashboard.get_alert_status()
        
        assert 'active_alerts' in status
        assert 'critical_alerts' in status
        assert 'warning_alerts' in status
        assert 'info_alerts' in status
        assert 'recent_alerts' in status
        assert 'alert_channels' in status
        assert 'last_updated' in status
        
        assert isinstance(status['recent_alerts'], list)
        assert isinstance(status['alert_channels'], dict)
    
    def test_rollback_history(self, safety_dashboard):
        """Test rollback history retrieval."""
        history = safety_dashboard.get_rollback_history()
        
        assert 'recent_rollbacks' in history
        assert 'rollback_capabilities' in history
        assert 'rollback_triggers' in history
        assert 'last_updated' in history
        
        assert isinstance(history['recent_rollbacks'], list)
        assert isinstance(history['rollback_capabilities'], dict)
        assert isinstance(history['rollback_triggers'], dict)
    
    def test_system_health_summary(self, safety_dashboard):
        """Test system health summary."""
        summary = safety_dashboard.get_system_health_summary()
        
        assert 'overall_health_score' in summary
        assert 'health_status' in summary
        assert 'components' in summary
        assert 'last_updated' in summary
        
        assert 0 <= summary['overall_health_score'] <= 1
        assert summary['health_status'] in ['healthy', 'warning', 'critical']
        assert 'feature_flags' in summary['components']
        assert 'slo_metrics' in summary['components']
        assert 'alert_status' in summary['components']
        assert 'rollback_history' in summary['components']
    
    def test_toggle_feature_flag(self, safety_dashboard):
        """Test feature flag toggling."""
        with patch('config.feature_flags.get_feature_flag_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.enable_feature.return_value = True
            mock_manager.disable_feature.return_value = True
            mock_manager.rollback_feature.return_value = True
            mock_get_manager.return_value = mock_manager
            
            # Test enable
            result = safety_dashboard.toggle_feature_flag('test_flag', 'enable', rollout_percentage=50.0)
            assert result['success'] is True
            assert result['action'] == 'enable'
            assert result['flag_name'] == 'test_flag'
            
            # Test disable
            result = safety_dashboard.toggle_feature_flag('test_flag', 'disable')
            assert result['success'] is True
            assert result['action'] == 'disable'
            
            # Test rollback
            result = safety_dashboard.toggle_feature_flag('test_flag', 'rollback')
            assert result['success'] is True
            assert result['action'] == 'rollback'
    
    def test_metrics_dashboard_data(self, safety_dashboard):
        """Test metrics dashboard data retrieval."""
        data = safety_dashboard.get_metrics_dashboard_data()
        
        assert 'health_summary' in data
        assert 'feature_flags' in data
        assert 'slo_metrics' in data
        assert 'charts' in data
        assert 'last_updated' in data
        
        charts = data['charts']
        assert 'slo_trends' in charts
        assert 'feature_rollout' in charts
        assert 'alert_timeline' in charts
        assert 'rollback_frequency' in charts
        
        assert isinstance(charts['slo_trends'], list)
        assert isinstance(charts['feature_rollout'], list)
        assert isinstance(charts['alert_timeline'], list)
        assert isinstance(charts['rollback_frequency'], list)
    
    def test_safety_controls_functionality(self, safety_dashboard):
        """Test safety controls functionality."""
        # Test all major functionality
        flag_status = safety_dashboard.get_feature_flag_status()
        slo_metrics = safety_dashboard.get_slo_metrics()
        alert_status = safety_dashboard.get_alert_status()
        rollback_history = safety_dashboard.get_rollback_history()
        health_summary = safety_dashboard.get_system_health_summary()
        dashboard_data = safety_dashboard.get_metrics_dashboard_data()
        
        # Verify all methods return valid data
        assert isinstance(flag_status, dict)
        assert isinstance(slo_metrics, dict)
        assert isinstance(alert_status, dict)
        assert isinstance(rollback_history, dict)
        assert isinstance(health_summary, dict)
        assert isinstance(dashboard_data, dict)

class TestGlobalFunctions:
    """Test global convenience functions."""
    
    def test_feature_flag_global_functions(self):
        """Test global feature flag functions."""
        from config.feature_flags import (
            is_feature_enabled, enable_feature, disable_feature,
            rollback_feature, get_feature_status, update_feature_metrics
        )
        
        # Test with mocked manager
        with patch('config.feature_flags.get_feature_flag_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.is_enabled.return_value = True
            mock_manager.enable_feature.return_value = True
            mock_manager.disable_feature.return_value = True
            mock_manager.rollback_feature.return_value = True
            mock_manager.get_feature_status.return_value = Mock()
            mock_get_manager.return_value = mock_manager
            
            # Test functions
            assert is_feature_enabled('test_flag') is True
            assert enable_feature('test_flag', 50.0) is True
            assert disable_feature('test_flag') is True
            assert rollback_feature('test_flag') is True
            assert get_feature_status('test_flag') is not None
            
            # Test metrics update
            update_feature_metrics('test_flag', {'test': 'value'})
            mock_manager.update_metrics.assert_called_once_with('test_flag', {'test': 'value'})
    
    def test_safety_controls_global_functions(self):
        """Test global safety controls functions."""
        from dashboard.safety_controls import (
            get_feature_flag_status, get_slo_metrics, get_alert_status,
            get_rollback_history, get_system_health_summary,
            toggle_feature_flag, get_metrics_dashboard_data
        )
        
        # Test with mocked dashboard
        with patch('dashboard.safety_controls.get_safety_controls_dashboard') as mock_get_dashboard:
            mock_dashboard = Mock()
            mock_dashboard.get_feature_flag_status.return_value = {'test': 'value'}
            mock_dashboard.get_slo_metrics.return_value = {'test': 'value'}
            mock_dashboard.get_alert_status.return_value = {'test': 'value'}
            mock_dashboard.get_rollback_history.return_value = {'test': 'value'}
            mock_dashboard.get_system_health_summary.return_value = {'test': 'value'}
            mock_dashboard.toggle_feature_flag.return_value = {'success': True}
            mock_dashboard.get_metrics_dashboard_data.return_value = {'test': 'value'}
            mock_get_dashboard.return_value = mock_dashboard
            
            # Test functions
            assert get_feature_flag_status() == {'test': 'value'}
            assert get_slo_metrics() == {'test': 'value'}
            assert get_alert_status() == {'test': 'value'}
            assert get_rollback_history() == {'test': 'value'}
            assert get_system_health_summary() == {'test': 'value'}
            assert toggle_feature_flag('test_flag', 'enable') == {'success': True}
            assert get_metrics_dashboard_data() == {'test': 'value'}

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
